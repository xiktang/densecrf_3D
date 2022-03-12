/*
    Copyright (c) 2011, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cstdio>
#include <cmath>



#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string.h>
#include <fstream>
#include <dirent.h>
#include <fnmatch.h>

#include "matio.h"

#include "../libDenseCRF/densecrf.h"
#include "../libDenseCRF/util.h"


template <typename Dtype> enum matio_classes matio_class_map();
template <> enum matio_classes matio_class_map<float>() { return MAT_C_SINGLE; }
template <> enum matio_classes matio_class_map<double>() { return MAT_C_DOUBLE; }
template <> enum matio_classes matio_class_map<int>() { return MAT_C_INT32; }
template <> enum matio_classes matio_class_map<unsigned int>() { return MAT_C_UINT32; }

template <typename T>
void LoadMatFile(const std::string& fn, T*& data, const int row, const int col,
		 int* channel = NULL, bool do_ppm_format = false);

template <typename T>
void LoadMatFile(const std::string& fn, T*& data, const int row, const int col, 
		 int* channel, bool do_ppm_format) {
  mat_t *matfp;
  matfp = Mat_Open(fn.c_str(), MAT_ACC_RDONLY);
  if (matfp == NULL) {
    std::cerr << "Error opening MAT file " << fn;
  }

  // Read data
  matvar_t *matvar;
  matvar = Mat_VarReadInfo(matfp,"data");
  if (matvar == NULL) {
    std::cerr << "Field 'data' not present in MAT file " << fn << std::endl;
  }

  if (matvar->class_type != matio_class_map<T>()) {
    std::cerr << "Field 'data' must be of the right class (single/double) in MAT file " << fn << std::endl;
  }
  if (matvar->rank >= 4) {
    if (matvar->dims[3] != 1) {
      std::cerr << "Rank: " << matvar->rank << ". Field 'data' cannot have ndims > 3 in MAT file " << fn << std::endl;
    }
  }

  std::cerr << "Row " << matvar->dims[0] << " Col " <<matvar->dims[1] << " Dim " <<matvar->dims[2] << " Rank " << matvar->rank <<std::endl;

  int file_size = 1;
  int data_size = row * col;
  for (int k = 0; k < matvar->rank; ++k) {
    file_size *= matvar->dims[k];
    
    if (k > 1) {
      data_size *= matvar->dims[k];
    }
  }

  assert(data_size <= file_size);

  T* file_data = new T[file_size];
  data = new T[data_size];
  
  int ret = Mat_VarReadDataLinear(matfp, matvar, file_data, 0, 1, file_size);
  if (ret != 0) {
    std::cerr << "Error reading array 'data' from MAT file " << fn << std::endl;
  }

  // matvar->dims[0] : width
  // matvar->dims[1] : height
  int in_offset = matvar->dims[0] * matvar->dims[1];
  int in_ind, out_ind;
  int data_channel = static_cast<int>(matvar->dims[2]);

  // extract from file_data
  if (do_ppm_format) {
    int out_offset = col * data_channel;

    for (int c = 0; c < data_channel; ++c) {
      for (int m = 0; m < row; ++m) {
	for (int n = 0; n < col; ++n) {
	  out_ind = m * out_offset + n * data_channel + c;

	  // perform transpose of file_data
	  in_ind  = n + m * matvar->dims[0];  

	  // note the minus sign
	  data[out_ind] = -file_data[in_ind + c*in_offset];  
	}
      }
    }
  } else {
    int out_offset = row * col;

    for (int c = 0; c < data_channel; ++c) {
      for (int m = 0; m < row; ++m) {
	for (int n = 0; n < col; ++n) {
	  in_ind  = m + n * matvar->dims[0];
	  out_ind = m + n * row; 
	  data[out_ind + c*out_offset] = -file_data[in_ind + c*in_offset];	  
	}
      }
    }
  }

  if(channel != NULL) {
    *channel = data_channel;
  }  


  Mat_VarFree(matvar);
  Mat_Close(matfp);

  delete[] file_data;
}



// Store the colors we read, so that we can write them again.
int nColors = 0;
int colors[255];

unsigned int getColor( float * c ){
  return c[0] + 256*c[1] + 256*256*c[2];
}
void putColor( unsigned char * c, unsigned int cc ){
  c[0] = cc&0xff; c[1] = (cc>>8)&0xff; c[2] = (cc>>16)&0xff;
}
// Produce a color image from a bunch of labels
unsigned char * colorize( const short * map, int W, int H ){
  unsigned char * r = new unsigned char[ W*H*3 ];
  for( int k=0; k<W*H; k++ ){
    int c = colors[ map[k] ];
    putColor( r+3*k, c );
  }
  return r;
}

// Certainty that the groundtruth is correct
const float GT_PROB = 0.5;

// Simple classifier that is 50% certain that the annotation is correct
float * classify( float * im, int W, int H, int M ){
  const float u_energy = -log( 1.0f / M );
  const float n_energy = -log( (1.0f - GT_PROB) / (M-1) );
  const float p_energy = -log( GT_PROB );
  float * res = new float[W*H*M];
  for( int k=0; k<W*H; k++ ){
    // Map the color to a label
    int c = getColor( im + 3*k );
    int i;
    for( i=0;i<nColors && c!=colors[i]; i++ );
    if (c && i==nColors){
      if (i<M)
	colors[nColors++] = c;
      else
	c=0;
    }
		
    // Set the energy
    float * r = res + k*M;
    if (c){
      for( int j=0; j<M; j++ )
	r[j] = n_energy;
      r[i] = p_energy;
    }
    else{
      for( int j=0; j<M; j++ )
	r[j] = u_energy;
    }
  }
  return res;
}

int main( int argc, char* argv[]){
  if (argc<4){
    printf("Usage: %s image annotations output\n", argv[0] );
    return 1;
  }
  // Number of labels
  const int M = 3;
  // Load the color image and some crude annotations (which are used in a simple classifier)
  int W, H, GW, GH;

  int feat_channel;
  bool do_ppm_format;

  //unsigned char * im = readPPM( argv[1], W, H );

  float * im;
  LoadMatFile(argv[1], im, H, W, &feat_channel, do_ppm_format=false);

  if (!im){
    printf("Failed to load image!\n");
    return 1;
  }
//  float * anno = readPPM( argv[2], GW, GH );
  float * anno;
  LoadMatFile(argv[1], im, GH, GW, &feat_channel, do_ppm_format=false);

  if (!anno){
    printf("Failed to load annotations!\n");
    return 1;
  }
  if (W!=GW || H!=GH){
    printf("Annotation size doesn't match image!\n");
    return 1;
  }
	
  /////////// Put your own unary classifier here! ///////////
  float * unary = classify( anno, W, H, M );
  ///////////////////////////////////////////////////////////
	
  // Setup the CRF model
  DenseCRF2D crf(W, H, M);
  // Specify the unary potential as an array of size W*H*(#classes)
  // packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ... (row-order)
  crf.setUnaryEnergy( unary );
  // add a color independent term (feature = pixel location 0..W-1, 0..H-1)
  // x_stddev = 3
  // y_stddev = 3
  // weight = 3
  crf.addPairwiseGaussian( 3, 3, 3 );
  // add a color dependent term (feature = xyrgb)
  // x_stddev = 60
  // y_stddev = 60
  // r_stddev = g_stddev = b_stddev = 20
  // weight = 10
  crf.addPairwiseBilateral( 60, 60, 20, 20, 20, im, 10 );
	
  // Do map inference
  short * map = new short[W*H];
  crf.map(10, map);
	
  // Store the result
  unsigned char *res = colorize( map, W, H );
  writePPM( argv[3], W, H, res );
	
  delete[] im;
  delete[] anno;
  delete[] res;
  delete[] map;
  delete[] unary;
}
