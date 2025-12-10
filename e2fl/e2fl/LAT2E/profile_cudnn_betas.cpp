#include <cudnn.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct ConvConfig {
    int N, C_in, H, W;
    int C_out;
    int Kh, Kw;
    int pad_h, pad_w;
    int stride_h, stride_w;
    int dil_h, dil_w;
    int groups;
};

static std::string detect_layer_type(const ConvConfig &cfg) {
    if (cfg.Kh == 1 && cfg.Kw == 1) return "conv1x1";
    if (cfg.groups == cfg.C_in)     return "depthwise";
    if (cfg.Kh == 3 && cfg.Kw == 3) return "conv3x3";
    return "unknown";
}

double compute_conv_flops(const ConvConfig &cfg, int H_out, int W_out) {
    long long flops = (long long)cfg.N * H_out * W_out *
                      cfg.C_out * ((cfg.C_in / cfg.groups) * cfg.Kh * cfg.Kw);
    return (double)flops;
}

// ---------------- conv forward ----------------
double run_cudnn_forward(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t xDesc, const float *x,
    cudnnFilterDescriptor_t wDesc, const float *w,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t yDesc, float *y,
    cudnnConvolutionFwdAlgo_t algo,
    int iters = 50
) {
    size_t workspace_bytes = 0;
    cudnnGetConvolutionForwardWorkspaceSize(
        handle, xDesc, wDesc, convDesc, yDesc, algo, &workspace_bytes);

    void *workspace = nullptr;
    if (workspace_bytes > 0) cudaMalloc(&workspace, workspace_bytes);

    float alpha = 1.f, beta = 0.f;

    // warmup
    for (int i = 0; i < 5; i++) {
        cudnnConvolutionForward(
            handle, &alpha,
            xDesc, x,
            wDesc, w,
            convDesc, algo,
            workspace, workspace_bytes,
            &beta,
            yDesc, y
        );
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < iters; i++) {
        cudnnConvolutionForward(
            handle, &alpha,
            xDesc, x,
            wDesc, w,
            convDesc, algo,
            workspace, workspace_bytes,
            &beta,
            yDesc, y
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (workspace) cudaFree(workspace);

    return ms / iters;
}

// ---------------- conv backward-data ----------------
double run_cudnn_backward_data(
    cudnnHandle_t handle,
    cudnnFilterDescriptor_t wDesc, const float *w,
    cudnnTensorDescriptor_t dyDesc, const float *dy,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t dxDesc, float *dx,
    cudnnConvolutionBwdDataAlgo_t algo,
    int iters = 50
) {
    size_t workspace_bytes = 0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle, wDesc, dyDesc, convDesc, dxDesc, algo, &workspace_bytes);

    void *workspace = nullptr;
    if (workspace_bytes > 0) cudaMalloc(&workspace, workspace_bytes);

    float alpha = 1.f, beta = 0.f;

    for (int i = 0; i < 5; i++) {
        cudnnConvolutionBackwardData(
            handle, &alpha,
            wDesc, w,
            dyDesc, dy,
            convDesc, algo,
            workspace, workspace_bytes,
            &beta,
            dxDesc, dx
        );
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < iters; i++) {
        cudnnConvolutionBackwardData(
            handle, &alpha,
            wDesc, w,
            dyDesc, dy,
            convDesc, algo,
            workspace, workspace_bytes,
            &beta,
            dxDesc, dx
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (workspace) cudaFree(workspace);

    return ms / iters;
}

// ---------------- conv backward-filter ----------------
double run_cudnn_backward_filter(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t xDesc, const float *x,
    cudnnTensorDescriptor_t dyDesc, const float *dy,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnFilterDescriptor_t dwDesc, float *dw,
    cudnnConvolutionBwdFilterAlgo_t algo,
    int iters = 50
) {
    size_t workspace_bytes = 0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle, xDesc, dyDesc, convDesc, dwDesc, algo, &workspace_bytes);

    void *workspace = nullptr;
    if (workspace_bytes > 0) cudaMalloc(&workspace, workspace_bytes);

    float alpha = 1.f, beta = 0.f;

    for (int i = 0; i < 5; i++) {
        cudnnConvolutionBackwardFilter(
            handle, &alpha,
            xDesc, x,
            dyDesc, dy,
            convDesc, algo,
            workspace, workspace_bytes,
            &beta,
            dwDesc, dw
        );
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < iters; i++) {
        cudnnConvolutionBackwardFilter(
            handle, &alpha,
            xDesc, x,
            dyDesc, dy,
            convDesc, algo,
            workspace, workspace_bytes,
            &beta,
            dwDesc, dw
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (workspace) cudaFree(workspace);

    return ms / iters;
}

// =============================== MAIN ===============================
int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./profile_fine <device_name> <model_name>\n";
        return -1;
    }

    std::string device_name = argv[1];  // e.g., "jetson_orin_nx"
    std::string model_name  = argv[2];  // e.g., "mobilenet_v2"

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    // conv configs
    std::vector<ConvConfig> configs = {
        {8,  32, 56, 56,  64, 3,3, 1,1, 1,1, 1,1, 1},
        {8,  64, 28, 28, 128, 3,3, 1,1, 1,1, 1,1, 64}, // depthwise
        {8, 128, 28, 28, 128, 1,1, 0,0, 1,1, 1,1, 1},
    };

    std::vector<cudnnConvolutionFwdAlgo_t> fwd_algos = {
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    };
    std::vector<cudnnConvolutionBwdDataAlgo_t> bwd_data_algos = {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
    };
    std::vector<cudnnConvolutionBwdFilterAlgo_t> bwd_filt_algos = {
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD,
    };

    // JSON 두 개
    json out;      // fine (LATTE 호환)
    json out_full; // fine_full (layer-type breakdown 포함)

    out["mode"]       = "fine";
    out["key_fwd"]    = json::object();
    out["key_bwd"]    = json::object();
    out["non_fwd"]    = 0.0;          // scalar
    out["non_bwd"]    = 0.0;          // scalar

    out_full["mode"]    = "fine_full";
    out_full["key_fwd"] = json::object();
    out_full["key_bwd"] = json::object();
    out_full["non_fwd"] = json::object(); // layer-type별로 기록
    out_full["non_bwd"] = json::object(); // layer-type별로 기록

    // ---------------- conv profiling ----------------
    for (auto &cfg : configs) {
        std::string layer_type = detect_layer_type(cfg);

        cudnnTensorDescriptor_t xDesc, yDesc, dyDesc, dxDesc;
        cudnnFilterDescriptor_t wDesc, dwDesc;
        cudnnConvolutionDescriptor_t convDesc;

        cudnnCreateTensorDescriptor(&xDesc);
        cudnnCreateTensorDescriptor(&yDesc);
        cudnnCreateTensorDescriptor(&dyDesc);
        cudnnCreateTensorDescriptor(&dxDesc);
        cudnnCreateFilterDescriptor(&wDesc);
        cudnnCreateFilterDescriptor(&dwDesc);
        cudnnCreateConvolutionDescriptor(&convDesc);

        cudnnSetTensor4dDescriptor(
            xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            cfg.N, cfg.C_in, cfg.H, cfg.W
        );
        cudnnSetFilter4dDescriptor(
            wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
            cfg.C_out, cfg.C_in / cfg.groups, cfg.Kh, cfg.Kw
        );
        cudnnSetConvolution2dDescriptor(
            convDesc,
            cfg.pad_h, cfg.pad_w,
            cfg.stride_h, cfg.stride_w,
            cfg.dil_h, cfg.dil_w,
            CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT
        );
        cudnnSetConvolutionGroupCount(convDesc, cfg.groups);

        int N, C, H_out, W_out;
        cudnnGetConvolution2dForwardOutputDim(
            convDesc, xDesc, wDesc, &N, &C, &H_out, &W_out
        );

        cudnnSetTensor4dDescriptor(
            yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            N, C, H_out, W_out
        );
        cudnnSetTensor4dDescriptor(
            dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            N, C, H_out, W_out
        );
        cudnnSetTensor4dDescriptor(
            dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            cfg.N, cfg.C_in, cfg.H, cfg.W
        );
        cudnnSetFilter4dDescriptor(
            dwDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
            cfg.C_out, cfg.C_in / cfg.groups, cfg.Kh, cfg.Kw
        );

        size_t x_bytes = (size_t)cfg.N * cfg.C_in * cfg.H * cfg.W * sizeof(float);
        size_t w_bytes = (size_t)cfg.C_out * (cfg.C_in / cfg.groups) * cfg.Kh * cfg.Kw * sizeof(float);
        size_t y_bytes = (size_t)N * C * H_out * W_out * sizeof(float);

        float *x_d = nullptr, *w_d = nullptr, *y_d = nullptr;
        float *dy_d = nullptr, *dx_d = nullptr, *dw_d = nullptr;

        cudaMalloc(&x_d, x_bytes);
        cudaMalloc(&w_d, w_bytes);
        cudaMalloc(&y_d, y_bytes);
        cudaMalloc(&dy_d, y_bytes);
        cudaMalloc(&dx_d, x_bytes);
        cudaMalloc(&dw_d, w_bytes);

        cudaMemset(x_d,  0, x_bytes);
        cudaMemset(w_d,  0, w_bytes);
        cudaMemset(y_d,  0, y_bytes);
        cudaMemset(dy_d, 0, y_bytes);
        cudaMemset(dx_d, 0, x_bytes);
        cudaMemset(dw_d, 0, w_bytes);

        double flops = compute_conv_flops(cfg, H_out, W_out);

        // forward
        for (auto algo : fwd_algos) {
            double ms = run_cudnn_forward(
                handle, xDesc, x_d, wDesc, w_d,
                convDesc, yDesc, y_d, algo
            );
            if (ms <= 0) continue;
            std::string key = layer_type + "_fwd_algo" + std::to_string((int)algo);
            double beta = ms / flops;
            out["key_fwd"][key]      = beta;
            out_full["key_fwd"][key] = beta;
        }

        // backward filter
        for (auto algo : bwd_filt_algos) {
            double ms = run_cudnn_backward_filter(
                handle, xDesc, x_d, dyDesc, dy_d,
                convDesc, dwDesc, dw_d, algo
            );
            if (ms <= 0) continue;
            std::string key = layer_type + "_bwd_filter_algo" + std::to_string((int)algo);
            double beta = ms / flops;
            out["key_bwd"][key]      = beta;
            out_full["key_bwd"][key] = beta;
        }

        // backward data
        for (auto algo : bwd_data_algos) {
            double ms = run_cudnn_backward_data(
                handle, wDesc, w_d, dyDesc, dy_d,
                convDesc, dxDesc, dx_d, algo
            );
            if (ms <= 0) continue;
            std::string key = layer_type + "_bwd_data_algo" + std::to_string((int)algo);
            double beta = ms / flops;
            out["key_bwd"][key]      = beta;
            out_full["key_bwd"][key] = beta;
        }

        cudaFree(x_d);
        cudaFree(w_d);
        cudaFree(y_d);
        cudaFree(dy_d);
        cudaFree(dx_d);
        cudaFree(dw_d);

        cudnnDestroyTensorDescriptor(xDesc);
        cudnnDestroyTensorDescriptor(yDesc);
        cudnnDestroyTensorDescriptor(dyDesc);
        cudnnDestroyTensorDescriptor(dxDesc);
        cudnnDestroyFilterDescriptor(wDesc);
        cudnnDestroyFilterDescriptor(dwDesc);
        cudnnDestroyConvolutionDescriptor(convDesc);
    }

    // ---------------- non-conv profiling ----------------
    {
        int Nnc = 1, Cnc = 256, Hnc = 64, Wnc = 64;
        size_t nc_bytes = (size_t)Nnc * Cnc * Hnc * Wnc * sizeof(float);

        float *nc_a = nullptr, *nc_b = nullptr, *nc_c = nullptr;
        cudaMalloc(&nc_a, nc_bytes);
        cudaMalloc(&nc_b, nc_bytes);
        cudaMalloc(&nc_c, nc_bytes);
        cudaMemset(nc_a, 0, nc_bytes);
        cudaMemset(nc_b, 0, nc_bytes);
        cudaMemset(nc_c, 0, nc_bytes);

        cudnnTensorDescriptor_t ncDesc;
        cudnnCreateTensorDescriptor(&ncDesc);
        cudnnSetTensor4dDescriptor(
            ncDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            Nnc, Cnc, Hnc, Wnc
        );

        // activation descriptor (ReLU)
        cudnnActivationDescriptor_t actDesc;
        cudnnCreateActivationDescriptor(&actDesc);
        cudnnSetActivationDescriptor(
            actDesc,
            CUDNN_ACTIVATION_RELU,
            CUDNN_PROPAGATE_NAN,
            0.0
        );

        // BN parameter desc (scale/bias/mean/var)
        cudnnTensorDescriptor_t bnParamDesc;
        cudnnCreateTensorDescriptor(&bnParamDesc);
        cudnnSetTensor4dDescriptor(
            bnParamDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            1, Cnc, 1, 1
        );

        float *bnScale = nullptr, *bnBias = nullptr;
        float *bnMean  = nullptr, *bnVar  = nullptr;
        cudaMalloc(&bnScale, Cnc * sizeof(float));
        cudaMalloc(&bnBias,  Cnc * sizeof(float));
        cudaMalloc(&bnMean,  Cnc * sizeof(float));
        cudaMalloc(&bnVar,   Cnc * sizeof(float));
        cudaMemset(bnScale, 1, Cnc * sizeof(float));
        cudaMemset(bnBias,  0, Cnc * sizeof(float));
        cudaMemset(bnMean,  0, Cnc * sizeof(float));
        cudaMemset(bnVar,   1, Cnc * sizeof(float));

        // for backward: dBnScale, dBnBias
        float *dBnScale = nullptr, *dBnBias = nullptr;
        cudaMalloc(&dBnScale, Cnc * sizeof(float));
        cudaMalloc(&dBnBias,  Cnc * sizeof(float));
        cudaMemset(dBnScale, 0, Cnc * sizeof(float));
        cudaMemset(dBnBias,  0, Cnc * sizeof(float));

        auto time_nc = [&](auto func) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            for (int i = 0; i < 5; i++) func();

            cudaEventRecord(start);
            for (int i = 0; i < 50; i++) func();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.f;
            cudaEventElapsedTime(&ms, start, stop);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return ms / 50.0;
        };

        double nc_flops = (double)Nnc * Cnc * Hnc * Wnc;
        float alpha_nc = 1.f, beta_nc = 0.f;

        // --- forward: ReLU ---
        double ms_relu_fwd = time_nc([&]() {
            cudnnActivationForward(
                handle,
                actDesc,
                &alpha_nc,
                ncDesc, nc_a,
                &beta_nc,
                ncDesc, nc_c
            );
        });
        double beta_relu_fwd = ms_relu_fwd / nc_flops;

        // --- forward: AddTensor ---
        double ms_add_fwd = time_nc([&]() {
            cudnnAddTensor(
                handle,
                &alpha_nc,
                ncDesc, nc_a,
                &beta_nc,
                ncDesc, nc_c
            );
        });
        double beta_add_fwd = ms_add_fwd / nc_flops;

        // --- forward: BatchNorm Inference ---
        double ms_bn_fwd = time_nc([&]() {
            cudnnBatchNormalizationForwardInference(
                handle,
                CUDNN_BATCHNORM_SPATIAL,
                &alpha_nc,
                &beta_nc,
                ncDesc, nc_a,
                ncDesc, nc_c,
                bnParamDesc,
                bnScale, bnBias,
                bnMean, bnVar,
                1e-5
            );
        });
        double beta_bn_fwd = ms_bn_fwd / nc_flops;

        // LATTE-style non_fwd scalar (fine.json)
        out["non_fwd"] =
            (beta_relu_fwd + beta_add_fwd + beta_bn_fwd) / 3.0;

        // fine_full: layer-type별로 non_fwd 기록
        out_full["non_fwd"]["relu"] = beta_relu_fwd;
        out_full["non_fwd"]["add"]  = beta_add_fwd;
        out_full["non_fwd"]["bn"]   = beta_bn_fwd;

        // --- backward: ReLU ---
        double ms_relu_bwd = time_nc([&]() {
            cudnnActivationBackward(
                handle,
                actDesc,
                &alpha_nc,
                ncDesc, nc_c,   // y
                ncDesc, nc_c,   // dy
                ncDesc, nc_a,   // x
                &beta_nc,
                ncDesc, nc_a    // dx
            );
        });
        double beta_relu_bwd = ms_relu_bwd / nc_flops;

        // --- backward: BatchNorm ---
        double ms_bn_bwd = time_nc([&]() {
            cudnnBatchNormalizationBackward(
                handle,
                CUDNN_BATCHNORM_SPATIAL,
                &alpha_nc,      // alphaDataDiff
                &beta_nc,       // betaDataDiff
                &alpha_nc,      // alphaParamDiff
                &beta_nc,       // betaParamDiff
                ncDesc, nc_a,   // xDesc, x
                ncDesc, nc_b,   // dyDesc, dy
                ncDesc, nc_c,   // dxDesc, dx
                bnParamDesc,    // dBnScaleBiasDesc
                bnScale,        // bnScale
                dBnScale,       // dBnScale
                dBnBias,        // dBnBias
                1e-5,           // epsilon
                bnMean,         // savedMean
                bnVar           // savedInvVar
            );
        });
        double beta_bn_bwd = ms_bn_bwd / nc_flops;

        // LATTE-style non_bwd scalar
        out["non_bwd"] =
            (beta_relu_bwd + beta_bn_bwd) / 2.0;

        // fine_full: layer-type별 non_bwd 기록
        out_full["non_bwd"]["relu"] = beta_relu_bwd;
        out_full["non_bwd"]["bn"]   = beta_bn_bwd;

        // cleanup non-conv
        cudaFree(nc_a);
        cudaFree(nc_b);
        cudaFree(nc_c);
        cudaFree(bnScale);
        cudaFree(bnBias);
        cudaFree(bnMean);
        cudaFree(bnVar);
        cudaFree(dBnScale);
        cudaFree(dBnBias);

        cudnnDestroyTensorDescriptor(ncDesc);
        cudnnDestroyTensorDescriptor(bnParamDesc);
        cudnnDestroyActivationDescriptor(actDesc);
    }

    // ---------------- save JSON files ----------------
    const char *home = std::getenv("HOME");
    std::string base = home ? std::string(home) : std::string(".");

    std::string dir = base + "/EEFL/E2FL/predictor/profile/" + device_name;
    std::string cmd = "mkdir -p " + dir;
    std::system(cmd.c_str());

    std::string fine_path      = dir + "/" + model_name + "_betas_fine.json";
    std::string fine_full_path = dir + "/" + model_name + "_betas_fine_full.json";

    std::ofstream ofs1(fine_path);
    ofs1 << out.dump(4);
    ofs1.close();

    std::ofstream ofs2(fine_full_path);
    ofs2 << out_full.dump(4);
    ofs2.close();

    std::cout << "Saved: " << fine_path << std::endl;
    std::cout << "Saved: " << fine_full_path << std::endl;

    cudnnDestroy(handle);
    return 0;
}
