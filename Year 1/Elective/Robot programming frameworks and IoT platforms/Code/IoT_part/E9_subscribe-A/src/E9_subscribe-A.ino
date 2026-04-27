// This #include statement was automatically added by the Particle IDE.
#include <Grove_ChainableLED.h>

// Include Particle Device OS APIs
#include "Particle.h"

// Let Device OS manage the connection to the Particle Cloud
SYSTEM_MODE(AUTOMATIC);

// Run the application and system concurrently in separate threads
SYSTEM_THREAD(ENABLED);

// Show system, cloud connectivity, and application logs over USB
// View logs with CLI using 'particle serial monitor --follow'
SerialLogHandler logHandler(LOG_LEVEL_INFO);

#define NUM_LEDS  1
ChainableLED leds(D4, D5, NUM_LEDS);

float hue = 0.0;
int buzzer = D2;

String message="";

void setup() {
    leds.init();
    pinMode(buzzer, OUTPUT);
    digitalWrite(buzzer, LOW);
    leds.setColorHSB(0, hue, 0.0, 0.0);
    Particle.subscribe("ANGLE", toggleLedBuz, MY_DEVICES);
}

void loop() {

}

void toggleLedBuz(const char *event, const char *data) {
    delay(1000);
    message=data;
    if (message=="LOW"){
        hue = 0.2;
        leds.setColorHSB(0, hue, 1.0, 0.5);
    }
    else if (message=="MEDIUM"){
        hue = 0.5;
        leds.setColorHSB(0, hue, 1.0, 0.5);
        tone(buzzer,1000,500);
    }
    else if (message=="HIGH"){
        hue = 1.0;
        leds.setColorHSB(0, hue, 1.0, 0.5);
        tone(buzzer,2000,500);
    }
    else{
        hue = 0.8;
        leds.setColorHSB(0, hue, 1.0, 0.5);
    }
}

