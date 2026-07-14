#include "Arduino_BMI270_BMM150.h"
#include <ArduinoEigen.h>
#include <Servo.h>

using Eigen::Matrix;
using Eigen::Vector;

#define WAIT_TIME 12     // Loop 주기 (ms)

float KP = 0.3;
float KI = 0.004;
float KD = 0.009;  // 미분 항 사용 안 함

float alpha = 0.9;
// P = 0.4, D = 0.007 I = 0.002

float err_prev  = 0.0;
float err_int   = 0.0;
float dt = 0.012;
float d_err = 0.0;
#define SERVO_CENTER_ZERO 90  // 수평 기준 서보 각도
int SERVO_CENTER = SERVO_CENTER_ZERO;

Servo servo1;
float ax, ay, az, gx, gy, gz;
float psi = 0.0, theta = 0.0, phi = 0.0;
unsigned long previousMillis = 0;
unsigned long lastServoUpdate = 0;

typedef struct { float x, y, z; } sensor;
sensor mean_acc[300] = {}, mean_gyro[300] = {};
sensor var_acc = {}, var_gyro = {};

Vector<float, 5> x;
Matrix<float, 5, 5> P, Q, A;
Matrix<float, 2, 5> H;
Matrix<float, 2, 2> R;

float off_psi, off_theta;

void kalman_update() {
  IMU.readAcceleration(ax, ay, az);
  IMU.readGyroscope(gx, gy, gz);

  float a_psi = atan2(ax, sqrt(ay * ay + az * az));
  float a_theta = atan2(ay, sqrt(ax * ax + az * az));

  psi = x(0); theta = x(1);
  float p = gx - x(2), q = gy - x(3), r = gz - x(4);
  float sec2 = 1.0 / pow(cos(theta), 2);

  A(0, 0) = 1 + dt * (q * cos(psi) * tan(theta) - r * sin(psi) * tan(theta));
  A(0, 1) =     dt * (q * sin(psi) * sec2 + r * cos(psi) * sec2);
  A(0, 2) = -dt;
  A(1, 0) =     dt * (-q * sin(psi) - r * cos(psi));
  A(1, 1) = 1;
  A(1, 3) = -dt;

  Vector<float, 5> xdot;
  xdot.setZero();
  xdot(0) = p + q * sin(psi) * tan(theta) + r * cos(psi) * tan(theta);
  xdot(1) = q * cos(psi) - r * sin(psi);

  Vector<float, 5> xp = x + xdot * dt;
  Matrix<float, 5, 5> Pp = A * P * A.transpose() + Q;

  Matrix<float, 5, 2> Ht = H.transpose();
  Matrix<float, 2, 2> S = H * Pp * Ht + R;
  Matrix<float, 5, 2> K = Pp * Ht * S.inverse();

  Vector<float, 2> z;
  z << a_psi - off_psi, a_theta - off_theta;

  Vector<float, 2> dz = z - H * xp;

  x = xp + K * dz;
  P = Pp - K * H * Pp;
}
//  0.6        0.002       0.007
void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  servo1.attach(11);
  servo1.write(SERVO_CENTER_ZERO);  // 초기 수평

  int cnt = 0;
  while (cnt < 300) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(ax, ay, az);
      IMU.readGyroscope(gx, gy, gz);
      mean_acc[cnt] = {ax, ay, az};
      mean_gyro[cnt] = {gx, gy, gz};
      cnt++;
    }
  }

  sensor sum_acc = {}, sum_gyro = {};
  for (int i = 0; i < 300; i++) {
    sum_acc.x += mean_acc[i].x;
    sum_acc.y += mean_acc[i].y;
    sum_acc.z += mean_acc[i].z;
    sum_gyro.x += mean_gyro[i].x;
    sum_gyro.y += mean_gyro[i].y;
    sum_gyro.z += mean_gyro[i].z;
  }

  sum_acc.x /= 300; sum_acc.y /= 300; sum_acc.z /= 300;
  sum_gyro.x /= 300; sum_gyro.y /= 300; sum_gyro.z /= 300;


  for (int i = 0; i < 300; i++) {
    var_acc.x += pow(mean_acc[i].x - sum_acc.x, 2);
    var_acc.y += pow(mean_acc[i].y - sum_acc.y, 2);
    var_acc.z += pow(mean_acc[i].z - sum_acc.z, 2);
    var_gyro.x += pow(mean_gyro[i].x - sum_gyro.x, 2);
    var_gyro.y += pow(mean_gyro[i].y - sum_gyro.y, 2);
    var_gyro.z += pow(mean_gyro[i].z - sum_gyro.z, 2);
  }

  var_acc.x /= 299; var_acc.y /= 299; var_acc.z /= 299;
  var_gyro.x /= 299; var_gyro.y /= 299; var_gyro.z /= 299;

  off_psi = atan2(sum_acc.x, sqrt(sum_acc.y * sum_acc.y + sum_acc.z * sum_acc.z));
  off_theta = atan2(sum_acc.y, sqrt(sum_acc.x * sum_acc.x + sum_acc.z * sum_acc.z));

  x.setZero();
  x(2) = sum_gyro.x;
  x(3) = sum_gyro.y;
  x(4) = sum_gyro.z;
  P.setZero(); Q.setZero(); A.setZero(); R.setZero();

  P(0, 0) = 1.0; P(1, 1) = 1.0; P(2, 2) = 0.1; P(3, 3) = 0.1; P(4, 4) = 0.1;
  Q(0, 0) = 1.2 * var_gyro.x;
  Q(1, 1) = 1.320377 * var_gyro.y;
  Q(2, 2) = 0.001008;
  Q(3, 3) = 0.001081;
  Q(4, 4) = 1e-3;

  R(0, 0) = 3.000002 * var_acc.x;
  R(1, 1) = 4.090002 * var_acc.y;

  H << 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0;
}

void loop() {
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= WAIT_TIME) {
    previousMillis = currentMillis;

     // === 실시간 PID 튜닝 키 입력 처리 ===
    if (Serial.available()) {
      char input = Serial.read();
      if (input == 'P') {
        KP += 0.1;
        Serial.print("KP increased: "); Serial.println(KP, 3);
        delay(500);
      } else if (input == 'p') {
        KP -= 0.1;
        //if (KP < 0) KP = 0;
        Serial.print("KP decreased: "); Serial.println(KP, 3);
        delay(500);
      } else if (input == 'D') {
        KD += 0.001;
        Serial.print("KD increased: "); Serial.println(KD, 3);
        delay(500);
      } else if (input == 'd') {
        KD -= 0.001;
        //if (KD < 0) KD = 0;
        Serial.print("KD decreased: "); Serial.println(KD, 3);
        delay(500);
      }
      if (input == 'I') {
        KI += 0.001;
        Serial.print("KI increased: "); Serial.println(KI, 3);
        delay(500);
      } else if (input == 'i') {
        KI -= 0.001;
        //if (KP < 0) KI = 0;
        Serial.print("KI decreased: "); Serial.println(KI, 3);
        delay(500);
      }
      if (input == 'A') {
        alpha += 0.1;
        Serial.print("Alpha increased: "); Serial.println(alpha, 3);
        delay(500);
      } else if (input == 'a') {
        alpha -= 0.1;
        //if (KP < 0) KI = 0;
        Serial.print("Alpha decreased: "); Serial.println(alpha, 3);
        delay(500);
      }
    }

    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      kalman_update();

      float roll_deg = x(0) * 180.0 / PI;
      Serial.print("Roll (deg): ");
      Serial.println(roll_deg);

      if (millis() > 2000 && (currentMillis - lastServoUpdate >= 50)) {
        lastServoUpdate = currentMillis;

        float target = 0.0;
        float error = target - roll_deg;
        err_int += error * dt;
        d_err = alpha * d_err + (1 - alpha) * ((error - err_prev) / dt);
        err_prev = error;

        float u = -KP * error - KI * err_int - KD * d_err;

        float max_change = 30.0;
        u = constrain(u, -max_change, max_change);

        SERVO_CENTER = constrain(SERVO_CENTER + u, 0, 180);
        servo1.write(SERVO_CENTER);

        Serial.print("Servo value: ");
        Serial.println(SERVO_CENTER);
      }
    }
  }
}
