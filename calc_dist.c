/*
 * PROJ1-1: YOUR TASK A CODE HERE
 *
 * You MUST implement the calc_min_dist() function in this file.
 *
 * You do not need to implement/use the swap(), flip_horizontal(), transpose(), or rotate_ccw_90()
 * functions, but you may find them useful. Feel free to define additional helper functions.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "digit_rec.h"
#include "utils.h"
#include "limits.h"
#include <emmintrin.h>

/* Swaps the values pointed to by the pointers X and Y. */
void swap(float *x, float *y) {
    float temp = *x;
    *x = *y;
    *y = temp;
}

/* Flips the elements of a square array ARR across the y-axis. */
float *flip_horizontal(float *arr, int width, int height) {
   #pragma omp parallel
  {
  float *temp = (float*) malloc(width * height * sizeof(float));
  memset(temp, 0, width * height * sizeof(float));
 
  #pragma omp for
  for (int i = 0; i < width * height; i++) {
    temp[i] = arr[i];
  }
  #pragma omp for
  for (int y = 0; y < height; y++) {
    #pragma omp for
    for (int x = 0; x < (width / 2); x += 1) {
      swap(&temp[(y * width) + x], &temp[(y * width) + width - 1 - x]);
    }
  }
  return temp;
  }
}


/* Transposes the square array ARR. */
void transpose(float *arr, int width) {
    /* Optional function */
}

/* Rotates the square array ARR by 90 degrees counterclockwise. */
float *rotate_ccw_90(float *arr, int width, int height) 
{
   #pragma omp parallel
  {
   int offset = height-1; 
   float *rotate = (float*)malloc(width * height * sizeof(float));
   float *p;
   #pragma omp for
   for(int i =0; i < height; i++)
   {
     p = arr+(i * width);
     #pragma omp for
     for(int j= 0; j<width; j++)
     {
       rotate[(j * height)+ offset] = *p;
       p++;
     }
     offset --; 
   }
   return rotate;
 }
}

/** Calculates and returns the squared euclidean distance between TEMPLATE and IMAGE. */
float euclid_dist(float *image, 
			 float *template, int t_width) {
   #pragma omp parallel
  {
  float squared_bits = 0;
  float total = 0;
  float numtosquare = 0;
  #pragma omp for
  for (int i = 0; (i < (t_width * t_width)); i ++) {
    float imageint = image[i] - '0';
    float templateint = template[i] - '0';
    numtosquare  = (imageint - templateint);
    squared_bits = numtosquare * numtosquare;
    total += squared_bits;
  }
  return total;
  }
}

/** Returns the minimum euclidean distance out of all of the possible translations. */
float translate(float *image, int i_width, int i_height, float *template, int t_width) {
  #pragma omp parallel
  {
    int template_indices[t_width * t_width]; //make new array of size twid * twid. This array keeps track of all the indices in the template that we want to look at.
    #pragma omp for
    for (int x = 0; x < t_width; x ++) { //sets the first twid - 1 indices in the array. 
	template_indices[x] = x;
    }
    int index = t_width;//this is for inside the next loop, so you can keep track of where you are in template_indices
    #pragma omp for
    for (int y = 1; y < t_width; y++) {//keeps track of each row
      #pragma omp for
      for (int p = 0; p < t_width; p++) {//adds inside of each row
	template_indices[index] = p + (y * i_width);//still setting the original square in template indices.
	index ++;
      }
    }
    float smallest = UINT_MAX;//the smallest distance we've seen so far
    while (template_indices[(t_width * t_width) - 1] < (i_width * i_height)) {//This is where we actually get the numbers in the indices
      float newimage[t_width * t_width];//the square we are going to look at and compare our template to
      #pragma omp for
      for (int z = 0; z < (t_width * t_width); z ++) {
	int place = template_indices[z];
	newimage[z] = image[place];//setting the values in the square from the values in the image.
      }
      float currentdist = euclid_dist(newimage, template, t_width);//get the distance for this translation
      if (currentdist < smallest) {
	smallest = currentdist;//check if its smaller than what we've already seen
      }
      #pragma omp for
      for (int r = 0; r < (t_width * t_width); r ++) {
	float toadd = 1;
	if ((template_indices[t_width - 1] % i_width) == i_width) {
	  toadd = t_width;
	}
	float newindex = template_indices[r] + toadd;
	template_indices[r] = newindex;
      }
    }
    return smallest;
  }
}

/* Returns the squared Euclidean distance between TEMPLATE and IMAGE. The size of IMAGE
 * is I_WIDTH * I_HEIGHT, while TEMPLATE is square with side length T_WIDTH. The template
 * image should be flipped, rotated, and translated across IMAGE.
 */
float calc_min_dist(float *image, int i_width, int i_height, float *template, int t_width) {
  #pragma omp parallel
  {
  float* rotate90 = rotate_ccw_90(image, i_width, i_height);
  float rotate90num = translate(rotate90, i_width, i_height, template, t_width);
  float* rotate180 = rotate_ccw_90(rotate90, i_width, i_height);
  float rotate180num = translate(rotate180, i_width, i_height, template, t_width);
  float* rotate270 = rotate_ccw_90(rotate180, i_width, i_height);
  float rotate270num = translate(rotate270, i_width, i_height, template, t_width);
  float* horizontal = flip_horizontal(image, i_width, i_height);
  float horizontalnum = translate(horizontal, i_width, i_height, template, t_width);
  float normalnum = translate(image, i_width, i_height, template, t_width);
  float* flipand90 = rotate_ccw_90(horizontal, i_width, i_height);
  float flipand90num = translate(flipand90, i_width, i_height, template, t_width);
  float* flipand180 = rotate_ccw_90(flipand90, i_width, i_height);
  float flipand180num = translate(flipand180, i_width, i_height, template, t_width);
  float* flipand270 = rotate_ccw_90(flipand180, i_width, i_height);
  float flipand270num = translate(flipand270, i_width, i_height, template, t_width);
  float minimums[] = {normalnum, horizontalnum, rotate90num, rotate180num, rotate270num, flipand90num, flipand180num, flipand270num};
  float min = minimums[0];
  #pragma omp for
  for (int i = 0; i < 8; i++) {//array length
    if (minimums[i] < min) {
      min = minimums[i];
    }
  }
  return min;
  }
}
