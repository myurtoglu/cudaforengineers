#ifndef INTERACTIONS_H
#define INTERACTIONS_H
#define W 600
#define H 600
#define DELTA_P 0.1f
#define TITLE_STRING "Stability"
int sys = 0;
float param = 0.1f;
void keyboard(unsigned char key, int x, int y) {
  if (key == 27) exit(0);
  if (key == '0') sys = 0;
  if (key == '1') sys = 1;
  if (key == '2') sys = 2;
  glutPostRedisplay();
}

void handleSpecialKeypress(int key, int x, int y) {
  if (key == GLUT_KEY_DOWN) param -= DELTA_P;
  if (key == GLUT_KEY_UP) param += DELTA_P;
  glutPostRedisplay();
}

// no mouse interactions implemented for this app
void mouseMove(int x, int y) { return; }
void mouseDrag(int x, int y) { return; }

void printInstructions() {
  printf("Stability visualizer\n");
  printf("Use number keys to select system:\n");
  printf("\t0: linear oscillator: positive stiffness\n");
  printf("\t1: linear oscillator: negative stiffness\n");
  printf("\t2: van der Pol oscillator: nonlinear damping\n");
  printf("up/down arrow keys adjust parameter value\n\n");
  printf("Choose the van der Pol (sys=2)\n");
  printf("Keep up arrow key depressed and watch the show.\n");
}

#endif