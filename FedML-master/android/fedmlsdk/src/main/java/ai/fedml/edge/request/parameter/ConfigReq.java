package ai.fedml.edge.request.parameter;

import com.google.gson.annotations.SerializedName;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class ConfigReq {
    public static final String MQTT_CONFIG = "mqtt_config";
    public static final String S3_CONFIG = "s3_config";

    public static final String MLOPS_CONFIG = "ml_ops_config";

    @SerializedName("config_name")
    private final List<String> configName = Collections.unmodifiableList(Arrays.asList(MQTT_CONFIG, S3_CONFIG, MLOPS_CONFIG));

    @SerializedName("device_send_time")
    private Long deviceSendTime;

    public List<String> getConfigName() {
        return configName;
    }

    public void setDeviceSendTime(Long deviceSendTime) {
        this.deviceSendTime = deviceSendTime;
    }

    public Long getDeviceSendTime() {
        return deviceSendTime;
    }

    public ConfigReq() {

    }
}
