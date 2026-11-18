// ============================================================
// Highway traffic-light LEDs
// ============================================================

const byte HIGHWAY_RED_PIN = 2;
const byte HIGHWAY_YELLOW_PIN = 3;
const byte HIGHWAY_GREEN_PIN = 4;


// ============================================================
// City-road traffic-light LEDs
// ============================================================

const byte CITY_RED_PIN = 5;
const byte CITY_YELLOW_PIN = 6;
const byte CITY_GREEN_PIN = 7;


// ============================================================
// Communication settings
// Must match hardware_demo.py
// ============================================================

const unsigned long SERIAL_BAUD_RATE = 115200;


// If no valid command is received for this duration,
// both directions enter the all-red fail-safe state.
const unsigned long COMMAND_TIMEOUT_MS = 5000;

unsigned long lastValidCommandTime = 0;
bool timeoutApplied = false;


// ============================================================
// LED helpers
// ============================================================

void setHighwayLights(
  bool redOn,
  bool yellowOn,
  bool greenOn
) {
  digitalWrite(HIGHWAY_RED_PIN, redOn ? HIGH : LOW);
  digitalWrite(HIGHWAY_YELLOW_PIN, yellowOn ? HIGH : LOW);
  digitalWrite(HIGHWAY_GREEN_PIN, greenOn ? HIGH : LOW);
}


void setCityLights(
  bool redOn,
  bool yellowOn,
  bool greenOn
) {
  digitalWrite(CITY_RED_PIN, redOn ? HIGH : LOW);
  digitalWrite(CITY_YELLOW_PIN, yellowOn ? HIGH : LOW);
  digitalWrite(CITY_GREEN_PIN, greenOn ? HIGH : LOW);
}


void setAllRed() {
  setHighwayLights(true, false, false);
  setCityLights(true, false, false);
}


// ============================================================
// Traffic-light states
// ============================================================

void setHighwayGreen() {
  // Highway green, city road red
  setHighwayLights(false, false, true);
  setCityLights(true, false, false);
}


void setHighwayYellow() {
  // Highway yellow, city road red
  setHighwayLights(false, true, false);
  setCityLights(true, false, false);
}


void setCityGreen() {
  // Highway red, city road green
  setHighwayLights(true, false, false);
  setCityLights(false, false, true);
}


void setCityYellow() {
  // Highway red, city road yellow
  setHighwayLights(true, false, false);
  setCityLights(false, true, false);
}


// ============================================================
// Serial command parser
// ============================================================

bool applyCommand(String command) {
  command.trim();

  if (command == "H:G,C:R") {
    setHighwayGreen();
  }
  else if (command == "H:Y,C:R") {
    setHighwayYellow();
  }
  else if (command == "H:R,C:G") {
    setCityGreen();
  }
  else if (command == "H:R,C:Y") {
    setCityYellow();
  }
  else if (command == "H:R,C:R") {
    setAllRed();
  }
  else {
    return false;
  }

  return true;
}


// ============================================================
// Arduino setup
// ============================================================

void setup() {
  pinMode(HIGHWAY_RED_PIN, OUTPUT);
  pinMode(HIGHWAY_YELLOW_PIN, OUTPUT);
  pinMode(HIGHWAY_GREEN_PIN, OUTPUT);

  pinMode(CITY_RED_PIN, OUTPUT);
  pinMode(CITY_YELLOW_PIN, OUTPUT);
  pinMode(CITY_GREEN_PIN, OUTPUT);

  Serial.begin(SERIAL_BAUD_RATE);
  Serial.setTimeout(100);

  // Safe startup state.
  setAllRed();

  lastValidCommandTime = millis();

  Serial.println("ARDUINO_READY");
}


// ============================================================
// Main loop
// ============================================================

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    bool valid = applyCommand(command);

    if (valid) {
      lastValidCommandTime = millis();
      timeoutApplied = false;

      Serial.print("ACK:");
      Serial.println(command);
    }
    else {
      // Invalid input produces a safe all-red state.
      setAllRed();

      Serial.print("ERROR:");
      Serial.println(command);
    }
  }

  // Communication fail-safe.
  if (
    millis() - lastValidCommandTime
    > COMMAND_TIMEOUT_MS
  ) {
    if (!timeoutApplied) {
      setAllRed();
      Serial.println("FAILSAFE:TIMEOUT");
      timeoutApplied = true;
    }
  }
}