// This #include statement was automatically added by the Particle IDE.
#include <Grove_Air_quality_Sensor.h>

// Include Particle Device OS APIs
#include "Particle.h"

// Let Device OS manage the connection to the Particle Cloud
SYSTEM_MODE(AUTOMATIC);

// Run the application and system concurrently in separate threads
SYSTEM_THREAD(ENABLED);

// Show system, cloud connectivity, and application logs over USB
// View logs with CLI using 'particle serial monitor --follow'
SerialLogHandler logHandler(LOG_LEVEL_INFO);

#define AQS_PIN A2
AirQualitySensor aqSensor(AQS_PIN);


// setup() runs once, when the device is first turned on
void setup() {
  // Put initialization like pinMode and begin functions here
  
  
  if (aqSensor.init())
 {
   Serial.println("Air Quality Sensor ready.");
 }
 else
 {
   Serial.println("Air Quality Sensor ERROR!");
 }
 
}

// loop() runs over and over again, as quickly as it can execute.
void loop() {
  // The core of your code will likely live here.

  // Example: Publish event to cloud every 10 seconds. Uncomment the next 3 lines to try it!
  // Log.info("Sending Hello World to the cloud!");
  // Particle.publish("Hello world!");
  // delay( 10 * 1000 ); // milliseconds and blocking - see docs for more info!
  
  String quality = getAirQuality();
  Serial.printlnf("Air Quality: %s", quality.c_str());
  delay(500);
  
}

String getAirQuality()
{
 int quality = aqSensor.slope();
 String qual = "None";

 if (quality == AirQualitySensor::FORCE_SIGNAL)
 {
   qual = "Danger";
 }
 else if (quality == AirQualitySensor::HIGH_POLLUTION)
 {
   qual = "High Pollution";
 }
 else if (quality == AirQualitySensor::LOW_POLLUTION)
 {
   qual = "Low Pollution";
 }
 else if (quality == AirQualitySensor::FRESH_AIR)
 {
   qual = "Fresh Air";
 }

 return qual;
}

//https://docs.particle.io/quickstart/aqmk-project/