#ifndef INTERACTIONS_H
#define INTERACTIONS_H
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif
#include <vector_types.h>
#define W 600
#define H 600
#define DELTA 5 // pixel increment for arrow keys
#define NX 128
#define NY 128
#define NZ 128

int id = 1; // 0 = sphere, 1 = torus, 2 = block
int method = 2; // 0 = volumeRender, 1 = slice, 2 = raycast
const int3 volumeSize = { NX, NY, NZ }; // size of volumetric data grid
const float4 params = { NX / 4.f, NY / 6.f, NZ / 16.f, 1.f };
float *d_vol; // pointer to device array for storing volume data
float zs = NZ; // distance from origin to source
float dist = 0.f, theta = 0.f, threshold = 0.f;

void mymenu(int value) {
  switch (value) {
  case 0: return;
  case 1: id = 0; break; // sphere
  case 2: id = 1; break; // torus
  case 3: id = 2; break; // block
  }
  volumeKernelLauncher(d_vol, volumeSize, id, params);
  glutPostRedisplay();
}

void createMenu() {
  glutCreateMenu(mymenu); // Object selection menu
  glutAddMenuEntry("Object Selector", 0); // menu title
  glutAddMenuEntry("Sphere", 1); // id = 1 -> sphere
  glutAddMenuEntry("Torus", 2); // id = 2 -> torus
  glutAddMenuEntry("Block", 3); // id = 3 -> block
  glutAttachMenu(GLUT_RIGHT_BUTTON); // right-click for menu
}

void keyboard(unsigned char key, int x, int y) {
  if (key == '+') zs -= DELTA; // move source closer (zoom in)
  if (key == '-') zs += DELTA; // move source farther (zoom out)
  if (key == 'd') --dist; // decrease slice distance
  if (key == 'D') ++dist; // increase slice distance
  if (key == 'z') zs = NZ, theta = 0.f, dist = 0.f; // reset values
  if (key == 'v') method = 0; // volume render
  if (key == 'f') method = 1; // slice
  if (key == 'r') method = 2; // raycast
  if (key == 27) exit(0);
  glutPostRedisplay();
}

void handleSpecialKeypress(int key, int x, int y) {
  if (key == GLUT_KEY_LEFT) theta -= 0.1f; // rotate left
  if (key == GLUT_KEY_RIGHT) theta += 0.1f; // rotate right
  if (key == GLUT_KEY_UP) threshold += 0.1f; // inc threshold (thick)
  if (key == GLUT_KEY_DOWN) threshold -= 0.1f; // dec threshold (thin)
  glutPostRedisplay();
}

void printInstructions() {
  printf("3D Volume Visualizer\n"
         "Controls:\n"
         "Volume render mode                          : v\n"
         "Slice render mode                           : f\n"
         "Raycast mode                                : r\n"
         "Zoom out/in                                 : -/+\n"
         "Rotate view                                 : left/right\n"
         "Decr./Incr. Offset (intensity in slice mode): down/up\n"
         "Decr./Incr. distance (only in slice mode)   : d/D\n"
         "Reset parameters                            : z\n"
         "Right-click for object selection menu\n");
}

#endif