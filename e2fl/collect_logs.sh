#!/bin/bash
# E2FL 로그 수집 스크립트

echo "🔄 E2FL 로그 수집 시작..."

# 결과 디렉토리 생성
RESULTS_DIR="results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
cd "$RESULTS_DIR"

echo "📁 결과 디렉토리: $RESULTS_DIR"

# 로컬 CSV 파일들 복사
echo "🔄 로컬 로그 수집 중..."
cp ../fl_*.csv . 2>/dev/null && echo "  ✅ 로컬 CSV 파일들 복사 완료" || echo "  ⚠️ 로컬 CSV 파일 없음"

# 각 클라이언트에서 CSV 파일 수집
echo "🔄 원격 클라이언트 로그 수집 중..."

# RPi5 클라이언트들
for i in 19 20 21 22 23; do
    echo "  📥 RPi5_$i (192.168.0.$i)에서 수집 중..."
    scp ubuntu@192.168.0.$i:~/EEFL/E2FL/fl_*.csv . 2>/dev/null && echo "    ✅ RPi5_$i 수집 완료" || echo "    ❌ RPi5_$i 연결 실패"
done

# Jetson 클라이언트
echo "  📥 Jetson (192.168.0.24)에서 수집 중..."
scp ubuntu@192.168.0.24:~/EEFL/E2FL/fl_*.csv . 2>/dev/null && echo "    ✅ Jetson 수집 완료" || echo "    ❌ Jetson 연결 실패"

# 수집된 파일들 확인
echo ""
echo "📊 수집된 파일들:"
ls -la fl_*.csv 2>/dev/null || echo "  ❌ CSV 파일이 없습니다."

# 분석 실행
echo ""
echo "📈 분석 실행 중..."
cd ..
python analyze_results.py

echo ""
echo "✅ 로그 수집 및 분석 완료!"
echo "📁 결과 위치: $RESULTS_DIR"
