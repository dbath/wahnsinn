
// ENTER EXPERIMENT INFORMATION HERE:
unsigned long PRE_STIM_PERIOD = 2000;           // enter the time before first stimulus (milliseconds)
unsigned long STIM_BOUT_DURATION = 2000;           // enter the duration of stimulus bouts (milliseconds)
unsigned long STIM_FREQUENCY = 4;                  //enter the frequency of stimulus (Hertz)
unsigned long PULSE_WIDTH = 125;            //enter stimulus pulse width (milliseconds)
unsigned long RECOVER_BOUT_DURATION = 1000;    //enter the duration of inter-stimulus recovery periods (milliseconds)
unsigned long POST_STIM_PERIOD = 3000;          //enter the duration of post-stimulus observation (milliseconds)
unsigned long STIM_BOUT_NUM = 5;                // enter the number of stimulus bouts



//--------------------DO NOT CHANGE ANYTHING BELOW THIS LINE----------------------------------------------

unsigned long PERIOD = 1000/STIM_FREQUENCY;
unsigned long OFF_PULSE = PERIOD - PULSE_WIDTH;
unsigned long COMPAT_TEST = STIM_BOUT_DURATION % PERIOD;

const int stimPin =  4;      // stimulus: on (pulsed) during stimulus bouts
const int indPin =  5;      // indicator: on (constant) during stimulus bouts
const int progPin =  6;      // program: on when program is engaged
const int initPin =  7;      // initializer: high when button is pressed

//GLOBAL VARIABLES:
boolean lastReading;  
boolean buttonState = LOW;
boolean lastButtonState;


//------------------------------------------------------------------------------------------------------------

void setup()
{
  Serial.begin(9600);
  pinMode(stimPin, OUTPUT);  
  pinMode(indPin, OUTPUT); 
  pinMode(progPin, OUTPUT); 
  pinMode(initPin, INPUT); 
}

void programRun(unsigned long initTime) {
  
  digitalWrite(progPin, HIGH);
  delay(PRE_STIM_PERIOD);
  for(int x = 0; x < STIM_BOUT_NUM; x++) {
    pulseTrain(millis());
    delay(RECOVER_BOUT_DURATION);
    x += 1;
  }
  delay(POST_STIM_PERIOD);
  digitalWrite(progPin, LOW);
}

void pulseTrain(unsigned long pulseBegin) {
  digitalWrite(indPin, HIGH);
  while ((millis() - pulseBegin) <= STIM_BOUT_DURATION){
    digitalWrite(indPin, HIGH);
    delay(PULSE_WIDTH);
    digitalWrite(stimPin, LOW);
    delay(OFF_PULSE);
  }
  digitalWrite(indPin, LOW);
}  

void errorMsg(int time) {
  digitalWrite(progPin, LOW);
  digitalWrite(indPin, HIGH);
  digitalWrite(stimPin, LOW);
  delay(time/2);
  digitalWrite(progPin, LOW);
  digitalWrite(indPin, HIGH);
  delay(time/2);
}
  
void loop() {
  while (OFF_PULSE < 0) {
    errorMsg(1000);
  }
  while (COMPAT_TEST != 0) {
    errorMsg(250);
  }
  
  //READ BUTTON:
  int reading = digitalRead(initPin);
  if (reading != lastButtonState) {
    lastButtonState = reading;
    if (reading == HIGH) {
      programRun(millis());
    }
  }
}
