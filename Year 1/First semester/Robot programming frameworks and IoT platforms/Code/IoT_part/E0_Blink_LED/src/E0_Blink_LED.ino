// Include Particle Device OS APIs
#include "Particle.h"

// Let Device OS manage the connection to the Particle Cloud
SYSTEM_MODE(AUTOMATIC);

// Run the application and system concurrently in separate threads
SYSTEM_THREAD(ENABLED);

// Show system, cloud connectivity, and application logs over USB
// View logs with CLI using 'particle serial monitor --follow'
SerialLogHandler logHandler(LOG_LEVEL_INFO);


int led = D7; // Instead of writing D7 over and over again, we'll write led2
// This one is the little blue LED on your board. On the Photon it is next to D7, and on the Core it is next to the USB jack.

void setup() {
    
  //pinMode(led1, OUTPUT);
  pinMode(led, OUTPUT);

}

void loop() {
  // To blink the LED, first we'll turn it on...
  digitalWrite(led, HIGH);  //digitalWrite(led2, HIGH);

  // We'll leave it on for 1 second...
  delay(500);

  // Then we'll turn it off...
  digitalWrite(led, LOW);
  //digitalWrite(led2, LOW);

  // Wait 1 second...
  delay(500);
}

