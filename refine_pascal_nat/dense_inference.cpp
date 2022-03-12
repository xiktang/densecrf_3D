/*
  Copyright (C) 2016 Ecole Polytechnique Federale de Lausanne 
  Modified by Timur Bagautdinov (timur.bagautdinov@epfl.ch) 

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>

  This file incorporates work covered by the following copyright and  
  permission notice:  

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
#include "../util/Timer.h"

template <typename Dtype> enum matio_classes matio_class_map();
template <> enum matio_classes matio_class_map<float>() { return MAT_C_SINGLE; }
template <> enum matio_classes matio_class_map<double>() { return MAT_C_DOUBLE; }
template <> enum matio_classes matio_class_map<int>() { return MAT_C_INT32; }
template <> enum matio_classes matio_class_map<unsigned int>() { return MAT_C_UINT32; }

template <typename T>
void LoadMatFile(const std::string& fn, T*& data, int& row, int& col,
		 int* channel = NULL, bool do_ppm_format = false);

template <typename T>
void LoadMatFile(const std::string& fn, T*& data, int& row, int& col, 
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

  row = matvar->dims[1];
  col = matvar->dims[0];

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


template <typename T>
void Load3DMatFile(const std::string& fn, T*& data, int& row, int& col, int& thick, //modify
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
  if (matvar->rank >= 5) {
    if (matvar->dims[4] != 1) {     //modif+ t * matvar->dims[0] * matvar->dims[1]y
      std::cerr << "Rank: " << matvar->rank << ". Field 'data' cannot have ndims > 4 in MAT file " << fn << std::endl;
    }
  }

  std::cerr << "Row " << matvar->dims[0] << " Col " <<matvar->dims[1] << " Thick " <<matvar->dims[2] << " Dim " <<matvar->dims[3] << " Rank " << matvar->rank <<std::endl;  //modify

  row = matvar->dims[1];
  col = matvar->dims[0];
  thick = matvar->dims[2]; //modify

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
  int in_offset = matvar->dims[0] * matvar->dims[1] * matvar->dims[2];
  int in_ind, out_ind;
  int data_channel = static_cast<int>(matvar->dims[3]);    //modify

  // extract from file_data
  if (do_ppm_format) {
    int out_offset1 = col * data_channel;
    int out_offset2 = row * col * data_channel;

    for (int c = 0; c < data_channel; ++c) {
      for (int m = 0; m < row; ++m) {
	for (int n = 0; n < col; ++n) {
	   for(int t = 0; t<thick; ++t){   //modify
	  out_ind = t * out_offset2 + m * out_offset1 + n * data_channel + c; //modify

	  // perform transpose of file_data
	  in_ind  = n + m * matvar->dims[0] + t * matvar->dims[0] * matvar->dims[1];  //modify

	  // note the minus sign
	  data[out_ind] = -file_data[in_ind + c*in_offset];    //modify
	   }
	}
      }
    }
  } else {
    int out_offset = row * col * thick;  //modify

    for (int c = 0; c < data_channel; ++c) {
      for (int m = 0; m < row; ++m) {
	for (int n = 0; n < col; ++n) {
		for(int t = 0; t<thick; ++t){  //modify
	  in_ind  = m + n * matvar->dims[0] + t * matvar->dims[0] * matvar->dims[1];  //midify
	  out_ind = m + n * row + t * row * col;          //modify
	  data[out_ind + c*out_offset] = -file_data[in_ind + c*in_offset];    //modify
		}
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


template <typename T>
void LoadBinFile(std::string& fn, T*& data, 
      int* row = NULL, int* col = NULL, int* channel = NULL);

template <typename T>
void SaveBinFile(std::string& fn, T* data, 
      int row = 1, int col = 15, int channel = 1);

template <typename T>
void LoadBinFile(std::string& fn, T*& data, 
    int* row, int* col, int* channel) {
  //data.clear();

  std::ifstream ifs(fn.c_str(), std::ios_base::in | std::ios_base::binary);

  if(!ifs.is_open()) {
    std::cerr << "Fail to open " << fn << std::endl;
  }

  int num_row, num_col, num_channel;

  ifs.read((char*)&num_row, sizeof(int));
  ifs.read((char*)&num_col, sizeof(int));
  ifs.read((char*)&num_channel, sizeof(int));

  int num_el;

  num_el = num_row * num_col * num_channel;

  //data.resize(num_el);
  data = new T[num_el];

  ifs.read((char*)&data[0], sizeof(T)*num_el);

  ifs.close();

  if(row!=NULL) {
    *row = num_row;
  }

  if(col!=NULL) {
    *col = num_col;
  }
 
  if(channel != NULL) {
    *channel = num_channel;
  }

}

template <typename T>
void SaveBinFile(std::string& fn, T* data, 
    int row, int col, int thick, int channel) {
  std::ofstream ofs(fn.c_str(), std::ios_base::out | std::ios_base::binary);

  if(!ofs.is_open()) {
    std::cerr << "Fail to open " << fn << std::endl;
  }  

  ofs.write((char*)&row, sizeof(int));
  ofs.write((char*)&col, sizeof(int));
  ofs.write((char*)&thick, sizeof(int));
  ofs.write((char*)&channel, sizeof(int));

  int num_el;

  num_el = row * col * thick * channel;

  ofs.write((char*)&data[0], sizeof(T)*num_el);

  ofs.close();
}

void SaveVector(const std::string &path, const std::vector<double> &v)
{
  std::ofstream ofs(path.c_str());
  ofs << v.size() << std::endl;
  for (size_t i = 0; i < v.size(); ++i)
    ofs << v[i] << " ";
}

void ReadCoocPotentials(const std::string &path,
			std::vector<float> &unary,
			std::vector<float> &pairwise,
			std::vector<float> &counts,
			int n_labels = 3,
			float min_cooc = 1.0)
{
  // TODO: a thing to do is to discretize
  // pairwise interactions into a fixed number of bins
  
  using namespace std;

  ifstream ifs(path.c_str(), ios::in);

  float n_cooc;

  ifs >> n_cooc;

  float value = 0.0;
  unary.resize(n_labels);
  for (size_t l = 0; l < n_labels; ++l)
  {
    ifs >> unary[l];
    unary[l] = (unary[l] + min_cooc) / n_cooc;
  }

  pairwise.resize(n_labels * n_labels);
  for (size_t l_a = 0; l_a < n_labels; ++l_a)
  {
    for (size_t l_b = 0; l_b < n_labels; ++l_b)
    {
      ifs >> pairwise[l_a*n_labels+l_b];
      pairwise[l_a*n_labels+l_b] += min_cooc;      
      pairwise[l_a*n_labels+l_b] /= (n_cooc + min_cooc);
    }
  }

  counts.resize(n_labels);
  for (size_t l = 0; l < n_labels; ++l)
    ifs >> counts[l];
}


void TraverseDirectory(const std::string& path, std::string& pattern,
		       bool subdirectories, std::vector<std::string>& fileNames)
{
  DIR *dir, *tstdp;
  struct dirent *dp;

  //open the directory
  if((dir  = opendir(path.c_str())) == NULL) {
    std::cout << "Error opening " << path << std::endl;
    return;
  }

  while ((dp = readdir(dir)) != NULL) {
    tstdp=opendir(dp->d_name);
		
    if(tstdp) {
      closedir(tstdp);
      if(subdirectories) {
	//TraverseDirectory(
      }
    } else {
      if(fnmatch(pattern.c_str(), dp->d_name, 0)==0) {
	//std::string tmp(path);	
	//tmp.append("/").append(dp->d_name);
	//fileNames.push_back(tmp);  //assume string ends with .bin

	std::string tmp(dp->d_name);
	fileNames.push_back(tmp.substr(0, tmp.length()-4));

	//std::cout << fileNames.back() << std::endl;
      }
    }
  }

  closedir(dir);
  return;
}


struct InputData {
  char* ImgDir;
  char* FeatureDir;
  char* SaveDir;
  int MaxIterations;
  float PosXStd;
  float PosYStd;
  float PosZStd;
  float PosW;
  float BilateralXStd;
  float BilateralYStd;
  float BilateralZStd;
  float BilateralRStd;
  float BilateralGStd;
  float BilateralBStd;
  float BilateralW;
  int FtrNum;
  char* Bibwname;
  char* Bipwname;
  char* BiXname;
  char* BiRname;

  // parameters for ngd
  //DenseCRF2DGD::InferenceType InferenceType;
  DenseCRF3DGD::InferenceType InferenceType;
  double StepSize;
  double MomentumGamma;

  // cooccurrence
  std::string CoocPath;
  float CoocStep;
  float CoocFactor;
};

int ParseInput(int argc, char** argv, struct InputData& OD) {
  for(int k=1;k<argc;++k) {
    if(::strcmp(argv[k], "-id")==0 && k+1!=argc) {
      OD.ImgDir = argv[++k];
    } else if(::strcmp(argv[k], "-fd")==0 && k+1!=argc) {
      OD.FeatureDir = argv[++k];
    } else if(::strcmp(argv[k], "-sd")==0 && k+1!=argc) {
      OD.SaveDir = argv[++k];
    } else if(::strcmp(argv[k], "-i")==0 && k+1!=argc) {
      OD.MaxIterations = atoi(argv[++k]);
    } else if(::strcmp(argv[k], "-px")==0 && k+1!=argc) {
      OD.PosXStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-py")==0 && k+1!=argc) {
      OD.PosYStd = atof(argv[++k]);
    }else if(::strcmp(argv[k], "-pz")==0 && k+1!=argc) {
      OD.PosZStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-pw")==0 && k+1!=argc) {
      OD.PosW = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-bx")==0 && k+1!=argc) {
      OD.BilateralXStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-by")==0 && k+1!=argc) {
      OD.BilateralYStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-bz")==0 && k+1!=argc) {
      OD.BilateralZStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-bw")==0 && k+1!=argc) {
      OD.BilateralW = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-br")==0 && k+1!=argc) {
      OD.BilateralRStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-bg")==0 && k+1!=argc) {
      OD.BilateralGStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-bb")==0 && k+1!=argc) {
      OD.BilateralBStd = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-type")==0 && k+1!=argc) {
      //OD.InferenceType = DenseCRF2DGD::InferenceType(atoi(argv[++k]));
    	OD.InferenceType = DenseCRF3DGD::InferenceType(atoi(argv[++k]));
    } else if(::strcmp(argv[k], "-step")==0 && k+1!=argc) {
      OD.StepSize = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-mg")==0 && k+1!=argc) {
      OD.MomentumGamma = atof(argv[++k]);
    } else if (::strcmp(argv[k], "-cooc_path")==0 && k+1!=argc) {
      OD.CoocPath = argv[++k];
    } else if (::strcmp(argv[k], "-cooc_step")==0 && k+1!=argc) {
      OD.CoocStep = atof(argv[++k]);
    } else if (::strcmp(argv[k], "-cooc_factor")==0 && k+1!=argc) {
      OD.CoocFactor = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-FtrNum")==0 && k+1!=argc) {
      OD.FtrNum = atof(argv[++k]);
    } else if(::strcmp(argv[k], "-bwnm")==0 && k+1!=argc) {
      OD.Bibwname = argv[++k];
    } else if(::strcmp(argv[k], "-pwnm")==0 && k+1!=argc) {
      OD.Bipwname = argv[++k];
    } else if(::strcmp(argv[k], "-Xnm")==0 && k+1!=argc) {
      OD.BiXname = argv[++k];
    } else if(::strcmp(argv[k], "-Rnm")==0 && k+1!=argc) {
      OD.BiRname = argv[++k];
    } 

  }
  return 0;
}

void ReshapeToMatlabFormat(short*& result, short* map, int img_row, int img_col) {
  //row-order to column-order

  int out_index, in_index;
  for (int h = 0; h < img_row; ++h) {
    for (int w = 0; w < img_col; ++w) {
      out_index = w * img_row + h;
      in_index  = h * img_col + w;
      result[out_index] = map[in_index];
    }
  }
}

void ReshapeToMatlabFormat3D(short*& result, short* map, int img_row, int img_col, int img_thick) {
  //row-order to column-order

  int out_index, in_index;
  for (int h = 0; h < img_row; ++h) {
    for (int w = 0; w < img_col; ++w) {
    	for (int l=0; l<img_thick; ++l){
          out_index = l * img_row * img_col + w * img_row + h;
          in_index  = l * img_row * img_col + h * img_col + w;
          result[out_index] = map[in_index];
    	}
    }
  }
}

void ComputeUnaryForCRF(float*& unary, float* feat, int feat_row, int feat_col, int feat_channel) {
  int out_index, in_index;

  for (int h = 0; h < feat_row; ++h) {
    for (int w = 0; w < feat_col; ++w) {
      for (int c = 0; c < feat_channel; ++c) {
	out_index = (h * feat_col + w) * feat_channel + c;
	in_index  = (c * feat_col + w) * feat_row + h;
	////unary[out_index] = -log(feat[in_index]);
	unary[out_index] = -feat[in_index];
      }
    }
  }
}

void GetImgNamesFromFeatFiles(std::vector<std::string>& out,
			      const std::vector<std::string>& in,
			      const std::string& strip_pattern)
{
  for (size_t k = 0; k < in.size(); ++k) {
    size_t pos = in[k].find(strip_pattern);
    if (pos != std::string::npos) {
      out.push_back(in[k].substr(0, pos));      
    }
  }
}

void OutputSetting(const InputData& inp)
{
  using namespace std;
  cout << string(80, '-') << endl;
  cout << "Input Parameters: " << endl;
  cout << "ImgDir:           " << inp.ImgDir << endl;
  cout << "FeatureDir:       " << inp.FeatureDir << endl;
  cout << "SaveDir:          " << inp.SaveDir << endl;
  cout << "MaxIterations:    " << inp.MaxIterations << endl;
  //cout << "MaxImgSize:       " << inp.MaxImgSize << endl;
  //cout << "NumClass:         " << inp.NumClass << endl;
  cout << "PosW:      " << inp.PosW    << endl;
  cout << "PosXStd:   " << inp.PosXStd << endl;
  cout << "PosYStd:   " << inp.PosYStd << endl;
  cout << "PosZStd:   " << inp.PosZStd << endl;
  cout << "Bi_W:      " << inp.BilateralW    << endl;
  cout << "Bi_X_Std:  " << inp.BilateralXStd << endl;
  cout << "Bi_Y_Std:  " << inp.BilateralYStd << endl;
  cout << "Bi_Z_Std:  " << inp.BilateralZStd << endl;
  cout << "Bi_R_Std:  " << inp.BilateralRStd << endl;
  cout << "Bi_G_Std:  " << inp.BilateralGStd << endl;
  cout << "Bi_B_Std:  " << inp.BilateralBStd << endl;
  cout << "Bibwname:   " << inp.Bibwname << endl;
  cout << "Bipwname:   " << inp.Bipwname << endl;
  cout << "BiXname:   " << inp.BiXname << endl;
  cout << "BiRname:   " << inp.BiRname << endl;
  

  cout << "FtrNum:  " << inp.FtrNum << endl;

  cout << "InferenceType: " << int(inp.InferenceType) << endl;
  cout << "StepSize: " << inp.StepSize << endl;
  cout << "MomentumGamma: " << inp.MomentumGamma << endl;  
  cout << "CoocPath: " << inp.CoocPath << endl;
  cout << "CoocStep: " << inp.CoocStep << endl;
  cout << "CoocFactor: " << inp.CoocFactor << endl;    
  cout << string(80, '-') << endl;  
  cout << "Hello farhad";	
}

int main(int argc, char* argv[])
{
  using namespace std;
  
  InputData inp;
  // default values
  inp.ImgDir = NULL;
  inp.FeatureDir = NULL;
  inp.SaveDir = NULL;
  inp.MaxIterations = 10;
  inp.BilateralW    = 5;
  inp.BilateralXStd = 70;
  inp.BilateralYStd = 70;
  inp.BilateralZStd = 70;    //modify
  inp.BilateralRStd = 5;
  inp.BilateralGStd = 5;
  inp.BilateralBStd = 5;
  inp.PosW    = 3;
  inp.PosXStd = 3;
  inp.PosYStd = 3;
  inp.PosZStd = 3;       //modify
  inp.Bibwname="";
  inp.Bipwname="";
  inp.BiXname="";
  inp.BiRname="";

  // GD stuff
  //inp.InferenceType = DenseCRF3DGD::INFERENCE_NAT; //DenseCRF2DGD::INFERENCE_NAT_ADAM; //DenseCRF2DGD::INFERENCE_NAT; //???
  inp.InferenceType = DenseCRF3DGD::INFERENCE_ADHOC; //DenseCRF3DGD::INFERENCE_NAT_ADAM;
  inp.StepSize = .1; //0.1; //.5
  inp.MomentumGamma = .95; //.95

  // Coocurence
  inp.CoocPath = "";
  inp.CoocStep = 1.0;
  inp.CoocFactor = 3.0;

  // Feature Number
  inp.FtrNum = 3;

  ParseInput(argc, argv, inp);
  OutputSetting(inp);

  assert(inp.ImgDir != NULL && inp.FeatureDir != NULL && inp.SaveDir != NULL);
  
  string pattern = "*_blob_0.mat";  //?????
  vector<string> feat_file_names;
  string feat_folder(inp.FeatureDir);

  TraverseDirectory(feat_folder, pattern, false, feat_file_names);
  
  string strip_pattern("_blob_0");
  vector<string> img_file_names;
  GetImgNamesFromFeatFiles(img_file_names, feat_file_names, strip_pattern);

  vector<float> cooc_unary, cooc_pairwise, cooc_counts;
  
  if (!inp.CoocPath.empty())
  {
    cout << string(80, '-') << endl;
    cout << "reading coocurrence potentials" << endl;
    ReadCoocPotentials(inp.CoocPath, cooc_unary, cooc_pairwise, cooc_counts);
    cout << "done!" << endl;
    cout << string(80, '-') << endl;  
  }

  CPrecisionTimer CTmr;
  CTmr.Start();

  cout << string(80, '-') << endl;
  cout << "starting inference for " << feat_file_names.size() << " images" << endl;

  #pragma omp parallel for num_threads(48)
  for (size_t i = 0; i < feat_file_names.size(); ++i)
  //for (int i = 0; i < 1; ++i)  
  {
    int feat_row, feat_col, feat_thick, feat_channel;
    bool do_ppm_format = true;

    //string fn = string(inp.ImgDir) + "/" + img_file_names[i] + ".ppm"; //?????
    string fn = string(inp.ImgDir) + "/" + img_file_names[i] + ".mat";
    cout << "image file name: " << fn << endl;	
    float* img;
    //unsigned char* img = readPPM(fn.c_str(), feat_col, feat_row);  // ?????
    //LoadMatFile(fn, img, feat_row, feat_col, &feat_channel, do_ppm_format=true);
    Load3DMatFile(fn, img, feat_row, feat_col, feat_thick, &feat_channel, do_ppm_format=true);   //modify
    cout << "Col: " << feat_col << " Row: " << feat_row << " Thickness: " << feat_thick << " Channel: "<< feat_channel <<endl;

    fn = string(inp.FeatureDir) + "/" + feat_file_names[i] + ".mat";
    cout << "feature file name: " << fn << endl;	
    float* feat;
    //LoadMatFile(fn, feat, feat_row, feat_col, &feat_channel, do_ppm_format=true);
    Load3DMatFile(fn, feat, feat_row, feat_col, feat_thick, &feat_channel, do_ppm_format=true);  //modify

    cout << "Mat file has been loaded." << endl;	
    // Setup the CRF model
    DenseCRF3DGD crf(feat_col, feat_row, feat_thick, feat_channel,                //modify
		     inp.InferenceType, inp.StepSize, inp.MomentumGamma);
    // Specify the unary potential as an array of size W*H*(#classes)
    // packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ... (row-order)
    crf.setUnaryEnergy(feat);
    // add a color independent term (feature = pixel location 0..W-1, 0..H-1)
    crf.addPairwiseGaussian(inp.PosXStd, inp.PosYStd, inp.PosZStd, inp.PosW);

    cout << "My print: " << inp.BilateralW << endl;	
    // add a color dependent term (feature = xyrgb)
    crf.addPairwiseBilateral(inp.BilateralXStd, inp.BilateralYStd, inp.BilateralZStd,
			     inp.BilateralRStd, inp.BilateralGStd, inp.BilateralBStd,
			     img, inp.BilateralW, inp.FtrNum);

    // add coocurence terms if necessary
    if (!inp.CoocPath.empty())
      crf.setCooc(cooc_unary, cooc_pairwise, cooc_counts, inp.CoocFactor, inp.CoocStep);
	
    // Do map inference
    short* map = new short[feat_row*feat_col*feat_thick];
    crf.map(inp.MaxIterations, map);

    short* result = new short[feat_row*feat_col*feat_thick];
    ReshapeToMatlabFormat3D(result, map, feat_row, feat_col, feat_thick);

    // save results
    fn = std::string(inp.SaveDir) + "/" + img_file_names[i] + "_Bibw_" + std::string(inp.Bibwname) + "_Bipw_" + std::string(inp.Bipwname) + "_BiX_" + std::string(inp.BiXname) + "_BiR_" + std::string(inp.BiRname) + ".bin";
    SaveBinFile(fn, result, feat_row, feat_col, feat_thick, 1);

    // saving kl and runtimes
SaveVector(std::string(inp.SaveDir) + "/" + img_file_names[i] + "_Bibw_" + std::string(inp.Bibwname) + "_Bipw_" + std::string(inp.Bipwname) + "_BiX_" + std::string(inp.BiXname) + "_BiR_" + std::string(inp.BiRname) + ".kl.txt", crf.kls());
    SaveVector(std::string(inp.SaveDir) + "/" + img_file_names[i] + "_Bibw_" + std::string(inp.Bibwname) + "_Bipw_" + std::string(inp.Bipwname) + "_BiX_" + std::string(inp.BiXname) + "_BiR_" + std::string(inp.BiRname) + ".runtimes.txt", crf.runtimes()); 
    
    // delete
    delete[] result;
    delete[] feat;
    delete[] img;
    delete[] map;

    cout << "done processing image " << i << endl;
  }
  cout << "Time for inference: " << CTmr.Stop() << endl;

  return 0;
}
