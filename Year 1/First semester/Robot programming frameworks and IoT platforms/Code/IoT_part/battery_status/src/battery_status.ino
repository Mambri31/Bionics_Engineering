// Include Particle Device OS APIs
#include "Particle.h"

// Let Device OS manage the connection to the Particle Cloud
SYSTEM_MODE(AUTOMATIC);

// Show system, cloud connectivity, and application logs over USB
// View logs with CLI using 'particle serial monitor --follow'
SerialLogHandler logHandler(LOG_LEVEL_INFO);

float voltage = 0;
bool charging = 0;

// setup() runs once, when the device is first turned on
void setup() {
  // Put initialization like pinMode and begin functions here
    Serial.begin(9600);

}

// loop() runs over and over again, as quickly as it can execute.
void loop() {
    
    voltage = analogRead(BATT) * 0.0011224;
    charging = digitalRead(CHG);
    
    
    Serial.println("VOLTAGE is: ");
    Serial.println(voltage);
    delay(1000);
    Serial.println("STATUS is; ");
    Serial.println(charging);
    delay(1000);
    
    //RGB.control(true);
    //RGB.color(255, 255, 255);
    //delay(1000);
    //RGB.control(false);

  // The core of your code will likely live here.

  // Example: Publish event to cloud every 10 seconds. Uncomment the next 3 lines to try it!
  // Log.info("Sending Hello World to the cloud!");
  // Particle.publish("Hello world!");
  // delay( 10 * 1000 ); // milliseconds and blocking - see docs for more info!
}