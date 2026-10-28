#include <Arduino_RouterBridge.h>
#include <Wire.h>

const uint8_t IMU_ADDR = 0x68;
const uint32_t PERIOD_US = 10000;   // 100 Hz
const int BATCH = 5;                // send every 5 samples (20 sends per second)

char buf[360];
int count = 0;
int len = 0;
uint32_t nextTick;

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(IMU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

int16_t read16() {
  uint8_t hi = Wire.read();
  uint8_t lo = Wire.read();
  return (int16_t)((hi << 8) | lo);
}

bool readSample(int16_t v[6]) {
  Wire.beginTransmission(IMU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(IMU_ADDR, (size_t)14) != 14) return false;
  v[0] = read16(); v[1] = read16(); v[2] = read16();   // ax ay az
  read16();                                            // temperature, skipped
  v[3] = read16(); v[4] = read16(); v[5] = read16();   // gx gy gz
  return true;
}

void setup() {
  Bridge.begin();
  Monitor.begin();
  Wire.begin();
  Wire.setClock(400000);
  delay(2000);

  writeReg(0x6B, 0x00);   // wake
  writeReg(0x1A, 0x03);   // low-pass about 44 Hz
  writeReg(0x1B, 0x10);   // gyro range +/-1000 deg/s
  writeReg(0x1C, 0x18);   // accel range +/-16 g

  nextTick = micros() + PERIOD_US;
}

void loop() {
  uint32_t now = micros();
  if ((int32_t)(now - nextTick) < 0) return;                // not time yet
  if ((int32_t)(now - nextTick) > 100000) nextTick = now;   // fell far behind, resync
  nextTick += PERIOD_US;

  int16_t v[6];
  if (!readSample(v)) return;

  len += snprintf(buf + len, sizeof(buf) - len, "%s%lu,%d,%d,%d,%d,%d,%d",
                  count ? ";" : "", (unsigned long)(now / 1000),
                  v[0], v[1], v[2], v[3], v[4], v[5]);
  count++;

  if (count >= BATCH) {
    Bridge.notify("imu", String(buf));
    count = 0; len = 0; buf[0] = 0;
  }
}
