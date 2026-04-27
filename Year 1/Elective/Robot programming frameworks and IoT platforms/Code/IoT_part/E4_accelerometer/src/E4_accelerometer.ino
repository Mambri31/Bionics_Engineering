// This #include statement was automatically added by the Particle IDE.
#include <OneWire.h>

// This #include statement was automatically added by the Particle IDE.
#include <Grove_IMU_9DOF_ICM20600_AK09918_y.h>
#include "AK09918.h"

// Include Particle Device OS APIs
#include "Particle.h"

// Let Device OS manage the connection to the Particle Cloud
SYSTEM_MODE(AUTOMATIC);

// Run the application and system concurrently in separate threads
SYSTEM_THREAD(ENABLED);

// Show system, cloud connectivity, and application logs over USB
// View logs with CLI using 'particle serial monitor --follow'
SerialLogHandler logHandler(LOG_LEVEL_INFO);

int32_t mx, my, mz;
AK09918 ak09918;
ICM20600 icm20600(true);
int16_t acc_x, acc_y, acc_z;

double roll, pitch;
// Find the magnetic declination at your location
// http://www.magnetic-declination.com/
double declination_pisa = -2.2;

void setup()
{
    // join I2C bus (I2Cdev library doesn't do this automatically)
    Wire.begin();

    icm20600.initialize();
    //ak09918.switchMode(AK09918_POWER_DOWN);
    ak09918.switchMode(AK09918_CONTINUOUS_100HZ);
    Serial.begin(9600);
}

void loop()
{
    // get acceleration
    acc_x = icm20600.getAccelerationX();
    acc_y = icm20600.getAccelerationY();
    acc_z = icm20600.getAccelerationZ();

    Serial.print("A:  ");
    Serial.print(acc_x);
    Serial.print(",  ");
    Serial.print(acc_y);
    Serial.print(",  ");
    Serial.print(acc_z);
    Serial.println(" mg");

    // get gyroscope
    Serial.print("G:  ");
    Serial.print(icm20600.getGyroscopeX());
    Serial.print(",  ");
    Serial.print(icm20600.getGyroscopeY());
    Serial.print(",  ");
    Serial.print(icm20600.getGyroscopeZ());
    Serial.println(" dps");

    // get magnetometer
    ak09918.getData(&mx, &my, &mz);

    Serial.print("M:  ");
    Serial.print(mx);
    Serial.print(",  ");
    Serial.print(my);
    Serial.print(",  ");
    Serial.print(mz);
    Serial.println(" uT");

    // roll/pitch in radian
    roll = atan2((float)acc_y, (float)acc_z);
    pitch = atan2(-(float)acc_x, sqrt((float)acc_y*acc_y+(float)acc_z*acc_z));
    Serial.print("Roll: ");
    Serial.println(roll*57.3);
    Serial.print("Pitch: ");
    Serial.println(pitch*57.3);

    double Xheading = mx * cos(pitch) + my * sin(roll) * sin(pitch) + mz * cos(roll) * sin(pitch);
    double Yheading = my * cos(roll) - mz * sin(pitch);
    

    double heading = 180 + 57.3*atan2(Yheading, Xheading) + declination_pisa;

    Serial.print("Heading: ");
    Serial.println(heading);
    Serial.println("--------------------------------");
  
    delay(500);
    
}
