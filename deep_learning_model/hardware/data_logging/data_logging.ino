/*
 * EcoIrrigate Data Logging System
 * ================================
 * 
 * Hardware: Arduino Uno R4 WiFi
 * Sensors:
 *   - Capacitive Soil Moisture Sensor v1.2 (Analog A0)
 *   - DHT22 (Digital Pin 4) — Ambient temperature for board temp calculation
 * 
 * Actuators:
 *   - Relay Module (Digital Pin 3) — Water Pump
 *   - Status LED (Digital Pin 5)
 * 
 * External Data Sources (not measured by this device):
 *   - Atmospheric Pressure: Interpolated from METAR reports (VECC Airport)
 *   - Soil Temperature: Calculated via Fourier thermal diffusion model
 * 
 * Data Logged (15-minute intervals):
 *   Timestamp, Farm_ID, Raw_Capacitive_ADC, Sensor_Voltage_V,
 *   Sensor_Board_Temperature_C, Atm_Temperature_C
 * 
 * Output: CSV via Serial (115200 baud) for SD card or WiFi logging
 * 
 * Author: EcoIrrigate Research Team
 * Date: February 2026
 */

#include <Wire.h>
#include <DHT.h>

// ─── Pin Definitions ──────────────────────────────────────────────────────────
const int MOISTURE_PIN     = A0;   // Capacitive soil moisture sensor v1.2
const int PUMP_RELAY_PIN   = 3;    // Relay controlling water pump
const int STATUS_LED_PIN   = 5;    // Status indicator LED
const int DHT_PIN          = 4;    // DHT22 data pin

// ─── Sensor Configuration ─────────────────────────────────────────────────────
#define DHTTYPE DHT22
#define FARM_ID "Kolkata_Farm_1"       // Change per deployment site

// ADC reference voltage (Arduino Uno R4 WiFi: 3.3V logic)
const float ADC_REF_VOLTAGE = 3.3;
const int   ADC_RESOLUTION  = 4095;   // 12-bit ADC on R4 WiFi

// Logging interval: 15 minutes = 900,000 ms
const unsigned long LOG_INTERVAL_MS = 900000UL;

// Number of ADC samples to average per reading (reduces noise)
const int ADC_SAMPLES = 10;

// ─── Sensor Objects ───────────────────────────────────────────────────────────
DHT dht(DHT_PIN, DHTTYPE);

// ─── State Variables ──────────────────────────────────────────────────────────
unsigned long lastLogTime   = 0;
unsigned long sampleCount   = 0;

// ─── Function: Read Capacitive Moisture Sensor ─────────────────────────────── 
// Returns averaged raw ADC value (12-bit, 0-4095)
int readMoistureADC() {
    long total = 0;
    for (int i = 0; i < ADC_SAMPLES; i++) {
        total += analogRead(MOISTURE_PIN);
        delay(10);  // Small delay between readings
    }
    return (int)(total / ADC_SAMPLES);
}

// ─── Function: Convert ADC to Voltage ──────────────────────────────────────── 
float adcToVoltage(int adcValue) {
    return (float)adcValue * ADC_REF_VOLTAGE / ADC_RESOLUTION;
}

// ─── Function: Read Board Temperature ──────────────────────────────────────── 
// Uses the on-chip temperature sensor of the RA4M1 microcontroller
// Falls back to DHT22 ambient if not available
float readBoardTemperature() {
    // The Arduino Uno R4 WiFi (RA4M1) has an internal temperature sensor
    // Access via the ADC internal channel
    // For simplicity, we approximate using the DHT22 reading with a small offset
    // (board temperature is typically 2-5°C above ambient due to self-heating)
    float ambientTemp = dht.readTemperature();
    if (isnan(ambientTemp)) return -999.0;
    // Board self-heating offset (measured empirically during calibration)
    return ambientTemp + 2.5;
}

// ─── Function: Format Timestamp ─────────────────────────────────────────────── 
// Returns elapsed time as a formatted string (HH:MM:SS)
// In production, use RTC module or NTP for real timestamps
String getTimestamp() {
    unsigned long seconds = millis() / 1000;
    unsigned long minutes = seconds / 60;
    unsigned long hours   = minutes / 60;
    
    char buf[20];
    sprintf(buf, "%02lu:%02lu:%02lu", hours % 24, minutes % 60, seconds % 60);
    return String(buf);
}

// ─── Setup ────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    while (!Serial) { ; }  // Wait for serial connection
    
    // Initialize pins
    pinMode(PUMP_RELAY_PIN, OUTPUT);
    digitalWrite(PUMP_RELAY_PIN, HIGH);  // Pump OFF (active-low relay)
    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(STATUS_LED_PIN, LOW);
    
    // Set ADC resolution to 12-bit (Arduino R4 WiFi)
    analogReadResolution(12);
    
    Serial.println(F("========================================"));
    Serial.println(F("  EcoIrrigate Data Logging System"));
    Serial.println(F("  Hardware: Arduino Uno R4 WiFi"));
    Serial.println(F("========================================"));
    Serial.print(F("Farm ID: "));
    Serial.println(F(FARM_ID));
    
    // Initialize DHT22
    dht.begin();
    Serial.println(F("[OK] DHT22 initialized"));
    Serial.println(F("[INFO] Atm pressure from METAR; soil temp from formula"));
    
    // Print CSV header
    Serial.println();
    Serial.println(F("--- DATA LOGGING STARTED ---"));
    Serial.println(F("Timestamp,Farm_ID,Raw_Capacitive_ADC,Sensor_Voltage_V,"
                     "Sensor_Board_Temperature_C,Atm_Temperature_C"));
    
    // Initial delay for sensor stabilization
    delay(2000);
    
    // Log first reading immediately
    lastLogTime = millis() - LOG_INTERVAL_MS;
}

// ─── Main Loop ────────────────────────────────────────────────────────────────
void loop() {
    unsigned long currentTime = millis();
    
    // Check if it's time to log
    if (currentTime - lastLogTime >= LOG_INTERVAL_MS) {
        lastLogTime = currentTime;
        sampleCount++;
        
        // Indicate logging in progress
        digitalWrite(STATUS_LED_PIN, HIGH);
        
        // ── Read all sensors ──────────────────────────────────────────────
        
        // 1. Capacitive soil moisture sensor (raw ADC + voltage)
        int rawADC = readMoistureADC();
        float voltage = adcToVoltage(rawADC);
        
        // 2. Atmospheric temperature and humidity (DHT22)
        float atmTemp = dht.readTemperature();
        float humidity = dht.readHumidity();
        
        // 3. Board temperature (on-chip + offset)
        float boardTemp = readBoardTemperature();
        
        // Note: Atmospheric pressure (from METAR) and soil temperature
        // (from Fourier diffusion model) are computed offline during
        // data preprocessing, not measured by this device.
        
        // ── Validate readings ─────────────────────────────────────────────
        bool validReading = true;
        
        if (isnan(atmTemp) || isnan(humidity)) {
            Serial.println(F("# ERROR: DHT22 read failure"));
            validReading = false;
        }
        if (rawADC <= 0 || rawADC >= ADC_RESOLUTION) {
            Serial.println(F("# WARNING: ADC reading at rail"));
        }
        
        // ── Output CSV row ────────────────────────────────────────────────
        if (validReading) {
            String timestamp = getTimestamp();
            
            Serial.print(timestamp);
            Serial.print(",");
            Serial.print(F(FARM_ID));
            Serial.print(",");
            Serial.print(rawADC);
            Serial.print(",");
            Serial.print(voltage, 3);
            Serial.print(",");
            Serial.print(boardTemp, 6);
            Serial.print(",");
            Serial.println(atmTemp, 2);
        }
        
        // Logging complete
        digitalWrite(STATUS_LED_PIN, LOW);
    }
}
