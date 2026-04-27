// This #include statement was automatically added by the Particle IDE.
#include <Adafruit_DHT_Particle.h>

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

#define DHTPIN 	D2
#define DHTTYPE DHT11
#define NUM_LEDS  1

DHT dht(DHTPIN, DHTTYPE);
ChainableLED leds(D4, D5, NUM_LEDS);

int currentTemp = 0;
int currentHum = 0;
float hue = 0.0;

unsigned long lastCheck = 0;


void setup() {

	dht.begin();
	leds.init();
	
	Particle.variable("temperature", currentTemp);
	Particle.variable("humidity", currentHum);
	Particle.function("turnOn", turnOnLED);
	
}

void loop() {
    unsigned long currentMillis = millis();
    
    if (currentMillis - lastCheck > 10000) {
        lastCheck = currentMillis;
        
        currentTemp = (int)dht.getTempCelcius();
        currentHum = (int)dht.getHumidity();
        
        /*if (currentHum >= 70) {
            hue = 0.0;
                leds.setColorHSB(0, hue, 1.0, 0.5);
        }
        
        if (currentHum < 70) {
            hue = 0.3;
                leds.setColorHSB(0, hue, 1.0, 0.5);
        }*/
    }
}


int turnOnLED(String args) {
    hue = args.toFloat();
    
    leds.setColorHSB(0, hue, 1.0, 0.5);
    
    return 1;
}


