

// Include Particle Device OS APIs
#include "Particle.h"
#include <Adafruit_GFX_RK.h>
#include <Adafruit_SSD1306.h>

struct VitalSigns
{
  int Spo2;
  int hr;
};

const VitalSigns patientData[] = {
  // --- NORMALE (0–60) ---
  {98,72},{98,73},{98,74},{97,73},{97,72},{98,74},{98,75},{97,74},{97,73},{98,72},
  {98,73},{98,74},{97,73},{97,72},{98,74},{98,75},{97,74},{97,73},{98,72},{98,73},
  {98,74},{97,73},{97,72},{98,74},{98,75},{97,74},{97,73},{98,72},{98,73},{98,74},
  {97,73},{97,72},{98,74},{98,75},{97,74},{97,73},{98,72},{98,73},{98,74},{97,73},
  {97,72},{98,74},{98,75},{97,74},{97,73},{98,72},{98,73},{98,74},{97,73},{97,72},
  {98,74},{98,75},{97,74},{97,73},{98,72},{98,73},{98,74},{97,73},{97,72},{98,74},

  // --- LOW ALERT (60–120) HR > 100 ---  SpO2 = 95
  {95,101}, {95,102}, {95,103}, {95,104}, {95,105}, {95,106}, {95,107}, {95,108}, {95,109}, {95,110},
  {95,111}, {95,112}, {95,113}, {95,114}, {95,115}, {95,116}, {95,117}, {95,118}, {95,119}, {95,120},
  {95,121}, {95,122}, {95,123}, {95,124}, {95,125}, {95,124}, {95,123}, {95,122}, {95,121}, {95,120},
  {95,119}, {95,118}, {95,117}, {95,116}, {95,115}, {95,114}, {95,113}, {95,112}, {95,111}, {95,110},
  {95,109}, {95,108}, {95,107}, {95,106}, {95,105}, {95,104}, {95,103}, {95,102}, {95,101}, {95,100},
  {95,101}, {95,102}, {95,103}, {95,104}, {95,105}, {95,106}, {95,107}, {95,108}, {95,109}, {95,110},


  // --- RIENTRO (120–170) ---
  {97,98},{97,96},{97,95},{97,94},{97,93},{97,92},{97,91},{97,90},{97,89},{97,88},
  {97,87},{97,86},{97,85},{97,84},{97,83},{97,82},{97,81},{97,80},{97,79},{97,78},
  {97,77},{97,76},{97,75},{97,74},{97,73},{97,72},{97,71},{97,70},{97,71},{97,72},
  {97,73},{97,74},{97,75},{97,76},{97,77},{97,78},{97,79},{97,80},{97,81},{97,82},
  {97,83},{97,84},{97,85},{97,86},{97,87},{97,88},{97,89},{97,90},{97,91},{97,92},

  // --- MEDIUM ALERT (170–230) ---
  {93,130},{93,132},{92,134},{92,136},{92,138},{92,140},{92,142},{92,144},{92,146},{92,148},
  {92,150},{92,148},{92,146},{92,144},{92,142},{92,140},{92,138},{92,136},{92,134},{92,132},
  {92,130},{92,132},{92,134},{92,136},{92,138},{92,140},{92,142},{92,144},{92,146},{92,148},
  {92,150},{92,152},{92,154},{92,156},{92,158},{92,160},{92,158},{92,156},{92,154},{92,152},
  {92,150},{92,148},{92,146},{92,144},{92,142},{92,140},{92,138},{92,136},{92,134},{92,132},
  {92,130},{92,128},{92,126},{92,124},{92,122},{92,120},{92,118},{92,116},{92,114},{92,112},

  // --- MAX ALERT (230–270) SpO2 < 90 ---
  {89,140},{88,142},{87,144},{86,146},{85,148},{84,150},{83,152},{82,154},{81,156},{80,158},
  {80,160},{80,162},{80,164},{80,166},{80,168},{80,170},{80,172},{80,174},{80,176},{80,178},
  {80,180},{81,178},{82,176},{83,174},{84,172},{85,170},{86,168},{87,166},{88,164},{89,162},
  {88,160},{87,158},{86,156},{85,154},{84,152},{83,150},{82,148},{81,146},{80,144},{80,142},

  // --- RECUPERO (270–320) ---
  {90,130},{91,125},{92,120},{93,115},{94,110},{95,105},{96,100},{97,95},{97,90},{97,85},
  {97,80},{97,78},{97,76},{97,74},{97,72},{98,72},{98,73},{98,74},{98,73},{98,72},
  {98,73},{98,74},{98,75},{98,74},{98,73},{98,72},{98,73},{98,74},{98,75},{98,74},
  {98,73},{98,72},{98,73},{98,74},{98,75},{98,74},{98,73},{98,72},{98,73},{98,74},
  {98,75},{98,74},{98,73},{98,72},{98,73},{98,74},{98,75},{98,74},{98,73},{98,72}
};



// Let Device OS manage the connection to the Particle Cloud
SYSTEM_MODE(AUTOMATIC);

// Show system, cloud connectivity, and application logs over USB
// View logs with CLI using 'particle serial monitor --follow'
SerialLogHandler logHandler(LOG_LEVEL_INFO);
// ---- OLED ----
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
// 
const uint8_t OLED_ADDR = 0x3C;

Adafruit_SSD1306 display(OLED_RESET);
const int numSamples = sizeof(patientData) / sizeof(patientData[0]);
int currentIndex = 0;
const int button_pin = D4;


void setup() {
  // Put initialization like pinMode and begin functions here
  Serial.begin(9600);
  // --- Button (D4) ---
  
  pinMode(button_pin, INPUT);

  display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  display.setTextColor(WHITE);
  display.setTextSize(1.5);
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("Starting...");
  display.display();
}
bool run = TRUE;
// loop() runs over and over again, as quickly as it can execute.
int alert_grade = 0;
double t_3 = 0; double t_2 = 0; double t_1 = 0;
double t_button = 100;
double last_time;
bool first_time = TRUE;
void loop() {
  if (!run) return;

  // 
  int button_state = digitalRead(button_pin);
  

  if (button_state == HIGH ) {
      double now = millis();
      if (now -last_time>90000 || first_time){
      first_time = FALSE;
      last_time = millis();
      const char* helpMsg = "The patient requested assistance.";
      Particle.publish("caregiver", helpMsg, PRIVATE);
      delay(200); 
    }
  }
  

  // ----- End of samples: stop BEFORE accessing patientData -----
  if (currentIndex >= numSamples) {
    run = false;
    Serial.println("Simulation finished");
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("Simulation finished");
    display.display();
    return;
  }

  int hr_i   = patientData[currentIndex].hr;
  int spo2_i = patientData[currentIndex].Spo2;
  double current_time_sec = millis() / 1000.0;

  int alert_grade = 0;

  // ---------- ALERT LOGIC ----------
  if (spo2_i <= 90) {                // Max alert
    alert_grade = 3;

    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("HIGH ALERT");
    display.println("Call emergency services");
    display.println();
    display.print("SpO2: "); display.println(spo2_i);
    display.print("HR:   "); display.println(hr_i);
    display.display();

    static double lastSent = 0;
    if (current_time_sec - lastSent > 90) {
      lastSent = current_time_sec;

      String msgCare =
        "<b>CRITICAL CONDITION<b>\n"
        "Please contact the medical unit immediately.\n"
        "Phone: 0522 679045";
      Particle.publish("caregiver", msgCare, PRIVATE);
      delay(1000);

      String msgMax =
        "<b>CRITICAL ALERT</b>\n"
        "Patient ID: 001456\n"
        "Vital signs are outside critical thresholds.\n"
        "<a href='https://stem.ubidots.com/app/dashboards/public/dashboard/"
        "ktlTg4XOZXHA109Tyhs5fTAjyMjba2hA7HwyZ1OkWWE?"
        "navbar=true&contextbar=false&layersBar=false'>Open live dashboard</a>\n";
      Particle.publish("max_alert_med", msgMax, PRIVATE);
      delay(1000);
    }
  }
  else if (spo2_i <= 93) {           // Medium alert
    alert_grade = 2;

    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("MEDIUM ALERT");
    display.println("Contact the caregiver");
    display.println();
    display.print("SpO2: "); display.println(spo2_i);
    display.print("HR:   "); display.println(hr_i);
    display.display();

    static double lastSent = 0;
    if (current_time_sec - lastSent > 90) {
      lastSent = current_time_sec;

      String msgMed =
        "<b>MEDIUM ALERT</b>\n"
        "Vital signs are outside the normal range.\n"
        "Close monitoring is recommended.\n"
        "<a href='https://stem.ubidots.com/app/dashboards/public/dashboard/"
        "ktlTg4XOZXHA109Tyhs5fTAjyMjba2hA7HwyZ1OkWWE?"
        "navbar=true&contextbar=false&layersBar=false'>View live parameters</a>";
      Particle.publish("caregiver", msgMed, PRIVATE);
      delay(1000);
    }
  }
  else if (spo2_i <= 95) {           // Low alert
    alert_grade = 1;

    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("LOW ALERT");
    display.println("Check recommended");
    display.println();
    display.print("SpO2: "); display.println(spo2_i);
    display.print("HR:   "); display.println(hr_i);
    display.display();
  }

  // ---------- NORMAL DISPLAY (only if no alert) ----------
  if (alert_grade == 0) {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.print("SpO2: "); display.println(spo2_i);
    display.print("HR:   "); display.println(hr_i);
    display.display();
  }

  // ---------- SAFE JSON PUBLISH ----------
  char buf[64];
  snprintf(buf, sizeof(buf), "{\"spo2\":%d, \"hr\":%d}", spo2_i, hr_i);

  // WITH_ACK supported; returns bool in classic usage
  bool ok = Particle.publish("vitals_data", buf, PRIVATE, WITH_ACK);
  (void)ok;

  currentIndex++;
  delay(1000);
}

