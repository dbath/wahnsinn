
// ENTER EXPERIMENT INFORMATION HERE:
long PRE_STIM_PERIOD = 5000;           // enter the time before first stimulus (milliseconds)
long STIM_BOUT_DURATION = 2000;           // enter the duration of stimulus bouts (milliseconds)
long STIM_FREQUENCY = 4;                  //enter the frequency of stimulus (Hertz)
long PULSE_WIDTH = 125;            //enter stimulus pulse width (milliseconds)
long RECOVER_BOUT_DURATION = 2000;    //enter the duration of inter-stimulus recovery periods (milliseconds)
long POST_STIM_PERIOD = 5000;          //enter the duration of post-stimulus observation (milliseconds)
long STIM_BOUT_NUM = 20;                // enter the number of stimulus bouts



//--------------------DO NOT CHANGE ANYTHING BELOW THIS LINE----------------------------------------------

const int stimPin1 =  0;      // stimulus 1: on (pulsed) during stimulus bouts on set 1
const int indPin1 =  1;      // indicator 1: on (constant) during stimulus bouts on set 1
const int progPin1 =  2;      // program 1: on when program is engaged on set 1
const int initPin1 =  3;      // initializer 1: high when button is pressed on set 1
const int stimPin2 =  4;      // stimulus 1: on (pulsed) during stimulus bouts on set 2
const int indPin2 =  5;      // indicator 1: on (constant) during stimulus bouts on set 2
const int progPin2 =  6;      // program 1: on when program is engaged on set 2
const int initPin2 =  7;      // initializer 1: high when button is pressed on set 2
const int stimPin3 =  8;      // stimulus 1: on (pulsed) during stimulus bouts on set 3
const int indPin3 =  9;      // indicator 1: on (constant) during stimulus bouts on set 3
const int progPin3 =  10;      // program 1: on when program is engaged on set 3
const int initPin3 =  11;      // initializer 1: high when button is pressed on set 3
const int test = 12;

PERIOD = 1000/FREQUENCY
//------------------------------------------------------------------------------------------------------------

void setup()
{
  Serial.begin(9600);
  
}

void loop()
{
  //do stuff, main
}

void buttonPress()
{
  //input: initTime = time when the button is pressed
  //input: initPin = the button that was pressed
  //output: expBegin = the time the experiment starts
  //output: stimPin = which pin to use as stimulus
  //output: indPin = which pin to use as indicator light
  //output: progPin = which pin to use as program indicator
}

void programRun(exp_begin, stimPin, indPin, progPin)
{
  //illuminate progPin
  //wait for pre-stim period
  //initiate a pulse train by calling pulseTrain(boutBegin)
  //repeat pulseTrain until stim bout num is reached
  //wait for post-stim period
  //turn off progPin
}

void pulseTrain(boutBegin, stimPin, indPin)
{
  
  // pulseCount == 0
  //if time is less than (boutBegin + STIM_BOUT_DURATION):
    //illuminate indPin
    //if time is less than boutBegin + PERIOD*pulseCount + PULSE_WIDTH:
      //turn on stimPin
    //else
      //turn off stimPin
      //increase pulseCount by 1
  //else
    //turn off indPin
    //return boutCount +=1
