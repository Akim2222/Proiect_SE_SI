int pinLED = 13;
boolean ledon;

void setup() {
  Serial.begin(9600);
  pinMode(pinLED, OUTPUT);
  ledon = true;
}
 
void loop() {
  Serial.print("Hello Pi\n");
  digitalWrite(pinLED, ledon = !ledon);
  delay(1000);
}
