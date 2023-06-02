#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <cstring>
#include <mpi.h>

using namespace cv;
using namespace std;

__global__ void haff(uchar* pixel, int rows, int cols)
{
     int j = blockIdx.x;
     int x = threadIdx.x;
     if (pow((cols/2 - x), 2) + pow((rows/2 - j), 2) <= pow(40, 2) && pow((cols/2 - x), 2) + pow((rows/2 - j), 2) > pow(39, 2) )//формулы круга
     {
         pixel[j * cols * 3 + x * 3] = 0;//синий
         pixel[j * cols * 3 + x * 3 + 1] = 255;//зеленый
         pixel[j * cols * 3 + x * 3 + 2] = 255;//красный
     }
}
__global__ void haff1(uchar* pixel, int rows, int cols)
{
    int j = blockIdx.x + rows/2;
    int x = threadIdx.x;
    if (pow((cols / 2 - x), 2) + pow((rows / 2 - j), 2) <= pow(40, 2) && pow((cols / 2 - x), 2) + pow((rows / 2 - j), 2) > pow(39, 2))//формулы круга
    {
        pixel[(j - rows/2) * cols * 3 + x * 3] = 0;//синий
        pixel[(j - rows/2) * cols * 3 + x * 3 + 1] = 255;//зеленый
        pixel[(j - rows/2) * cols * 3 + x * 3 + 2] = 255;//красный
    }
}

__global__ void fish_eye(uchar* pixel, uchar* pixel1, int rows, int cols)
{
    float N = (float)cols;
    float M = (float)rows;
    float omega = 0.7;
    float x0 = 0, y0 = 0, r0 = 0, teta = 0, scale = 0, r1 = 0;
    int x1, y1;
    float e = 0.01;

    int i = threadIdx.x;
    int y = blockIdx.x;

    x0 = (i - ((N + 1) / 2)) * 2 / N;

    y0 = (y - ((M + 1) / 2)) * 2 / M;

    r0 = sqrt(pow(x0, 2) + pow(y0, 2));

    teta = atan2(y0, x0);

    scale = min(1 / (abs(cos(teta) + e)), 1 / (abs(sin(teta) + e)));

    r1 = min(scale, (float)1.0) * pow(r0, omega);

    x1 = ((N / 2) * r1 * cos(teta) + ((N + 1) / 2));

    y1 = ((M / 2) * r1 * sin(teta) + ((M + 1) / 2));


    pixel1[y1 * cols * 3 + x1 * 3] = pixel[y * cols * 3 + i * 3];
    pixel1[y1 * cols * 3 + x1 * 3 + 1] = pixel[y * cols * 3 + i * 3 + 1];
    pixel1[y1 * cols * 3 + x1 * 3 + 2] = pixel[y * cols * 3 + i * 3 + 2];
}
__global__ void fish_eye1(uchar* pixel, uchar* pixel1, int rows, int cols)
{
    float N = (float)cols;
    float M = (float)rows;
    float omega = 0.7;
    float x0 = 0, y0 = 0, r0 = 0, teta = 0, scale = 0, r1 = 0;
    int x1, y1;
    float e = 0.01;

    int i = threadIdx.x;
    int y = blockIdx.x + rows/2;

    x0 = (i - ((N + 1) / 2)) * 2 / N;

    y0 = (y - ((M + 1) / 2)) * 2 / M;

    r0 = sqrt(pow(x0, 2) + pow(y0, 2));

    teta = atan2(y0, x0);

    scale = min(1 / (abs(cos(teta) + e)), 1 / (abs(sin(teta) + e)));

    r1 = min(scale, (float)1.0) * pow(r0, omega);///////////////////////////////////////////////////////////

    x1 = ((N / 2) * r1 * cos(teta) + ((N + 1) / 2));

    y1 = ((M / 2) * r1 * sin(teta) + ((M + 1) / 2));


    pixel1[y1 * cols * 3 + x1 * 3] = pixel[y * cols * 3 + i * 3];
    pixel1[y1 * cols * 3 + x1 * 3 + 1] = pixel[y * cols * 3 + i * 3 + 1];
    pixel1[y1 * cols * 3 + x1 * 3 + 2] = pixel[y * cols * 3 + i * 3 + 2];
}

__global__ void interpol(uchar* pixel1, uchar* pixel, int rows, int cols)
{

    float alpha_y;
    int ty0, ty1, ty;

    int y = blockIdx.x;
    int x = threadIdx.x * 3;

    if (pixel1[y * cols * 3 + x] == 255 && pixel1[y * cols * 3 + x + 1] == 255 && pixel1[y * cols * 3 + x + 2] == 255)
    {
        for (int j = 1; y + j < rows; j++)
        {
            if (pixel1[(y + j) * cols * 3 + x] != 255 && pixel1[(y + j) * cols * 3 + x + 1] != 255 && pixel1[(y + j) * cols * 3 + x + 2] != 255)
            {
                ty0 = y + j;
                break;
            }
        }
        for (int j = -1; y + j == 0; j--)
        {
            if (pixel1[(y + j) * cols * 3 + x] != 255 && pixel1[(y + j) * cols * 3 + x + 1] != 255 && pixel1[(y + j) * cols * 3 + x + 2] != 255)
            {
                ty1 = y - j;
                break;
            }
        }
        ty = y;
        alpha_y = (float)(ty1 - ty) / (float)(ty1 - ty0);


        ////answer
        pixel[y * cols * 3 + x] = pixel1[ty0 * cols * 3 + x] * alpha_y + (1 - alpha_y) * pixel1[ty1 * cols * 3 + x];
        pixel[y * cols * 3 + x + 1] = pixel1[ty0 * cols * 3 + x + 1] * alpha_y + (1 - alpha_y) * pixel1[ty1 * cols * 3 + x + 1];
        pixel[y * cols * 3 + x + 2] = pixel1[ty0 * cols * 3 + x + 2] * alpha_y + (1 - alpha_y) * pixel1[ty1 * cols * 3 + x + 2];
        if (x == cols/2 * 3)
        {
            pixel[y * cols * 3 + x] = pixel[y * cols * 3 + (x - 1)];
            pixel[y * cols * 3 + x + 1] = pixel[y * cols * 3 + (x - 1) + 1];
            pixel[y * cols * 3 + x + 2] = pixel[y * cols * 3 + (x - 1) + 2];
        }

    }


}
__global__ void interpol1(uchar* pixel1, uchar* pixel, int rows, int cols)
{

    float alpha_y;
    int ty0, ty1, ty;

    int y = blockIdx.x + rows/2;
    int x = threadIdx.x * 3;

    if (pixel1[y * cols * 3 + x] == 255 && pixel1[y * cols * 3 + x + 1] == 255 && pixel1[y * cols * 3 + x + 2] == 255)
    {
        for (int j = 1; y + j < rows; j++)
        {
            if (pixel1[(y + j) * cols * 3 + x] != 255 && pixel1[(y + j) * cols * 3 + x + 1] != 255 && pixel1[(y + j) * cols * 3 + x + 2] != 255)
            {
                ty0 = y + j;
                break;
            }
        }
        for (int j = -1; y + j == 0; j--)
        {
            if (pixel1[(y + j) * cols * 3 + x] != 255 && pixel1[(y + j) * cols * 3 + x + 1] != 255 && pixel1[(y + j) * cols * 3 + x + 2] != 255)
            {
                ty1 = y - j;
                break;
            }
        }
        ty = y;
        alpha_y = (float)(ty1 - ty) / (float)(ty1 - ty0);


        ////answer
        pixel[y * cols * 3 + x] = pixel1[ty0 * cols * 3 + x] * alpha_y + (1 - alpha_y) * pixel1[ty1 * cols * 3 + x];
        pixel[y * cols * 3 + x + 1] = pixel1[ty0 * cols * 3 + x + 1] * alpha_y + (1 - alpha_y) * pixel1[ty1 * cols * 3 + x + 1];
        pixel[y * cols * 3 + x + 2] = pixel1[ty0 * cols * 3 + x + 2] * alpha_y + (1 - alpha_y) * pixel1[ty1 * cols * 3 + x + 2];
        if (x == cols/2 * 3)
        {
            pixel[y * cols * 3 + x] = pixel[y * cols * 3 + (x - 1)];
            pixel[y * cols * 3 + x + 1] = pixel[y * cols * 3 + (x - 1) + 1];
            pixel[y * cols * 3 + x + 2] = pixel[y * cols * 3 + (x - 1) + 2];
        }

    }


}

int main(int argc, char* argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Mat image0 = imread("./cat.png");
    if (image0.empty())
    {
        std::cerr << "Could not load image";
        return 1;
    }
    Mat image1(image0.rows, image0.cols, CV_8UC3); //поле для копирования
    Mat image2(image0.rows, image0.cols, CV_8UC3); //поле для копирования
   
    cout << image0.rows << "x" << image0.cols << endl;// <---строки(rows) и столбцы(cols) соответственно (x,y)
    const int rows = image0.rows; // rows = 300
    const int cols = image0.cols; // cols = 400

    //рисуем окружность
    Point center(cols/2, rows/2);
    int r = 40;
    Scalar LineColor(0, 0, 255);
    int thickness = 1;
    circle(image0, center, r, LineColor, thickness);
    namedWindow("Image0");
    imshow("Image0", image0);
    //init
    MPI_Status status;

    uchar* pix;
    pix = new uchar[rows * cols * 3];
    uchar* pix_buf;
    pix_buf = new uchar[rows/2 * cols * 3];
    uchar* ex_pix_buf;
    ex_pix_buf = new uchar[rows * cols * 3];

    uchar* d_pix;
    uchar* d_ex_pix;

    uchar* pix1;
    pix1 = new uchar[rows * cols * 3];
    uchar* pix_buf1;
    pix_buf1 = new uchar[rows/2 * cols * 3];
    uchar* ex_pix_buf1;
    ex_pix_buf1 = new uchar[rows * cols * 3];

    uchar* d_pix1;
    uchar* d_ex_pix1;
    
    int fff = 0;
    //start_____________________________________________________________________________
    if (rank == 0)
    {
        //alloc memory on CUDA
        cudaMalloc((void**)&d_pix, rows/2 * cols * 3 * sizeof(uchar));
        cudaMalloc((void**)&d_pix1, rows/2 * cols * 3 * sizeof(uchar));
        cudaMalloc((void**)&d_ex_pix, rows * cols * 3 * sizeof(uchar));
        cudaMalloc((void**)&d_ex_pix1, rows * cols * 3 * sizeof(uchar));
        
        // img -> byte
        for (int y = 0; y < image0.rows; y++)
        {
            for (int x = 0; x < image0.cols; x++)
            {
                pix[fff] = image0.at<Vec3b>(y, x)[0];
                pix[fff + 1] = image0.at<Vec3b>(y, x)[1];
                pix[fff + 2] = image0.at<Vec3b>(y, x)[2];
                fff += 3;
            }
        }
        //copy in cuda
        cudaMemcpy(d_pix, pix, rows/2 * cols * 3 * sizeof(uchar), cudaMemcpyHostToDevice);
        //haff
        haff << <rows / 2, cols >> > (d_pix, rows, cols);
        //copy in host
        cudaMemcpy(pix, d_pix, rows/2 * cols * 3 * sizeof(uchar), cudaMemcpyDeviceToHost);
      
        memcpy(pix + rows/2 * cols * 3, pix_buf, rows / 2 * cols * 3);

        fff = 0;
        //byte -> img
        for (int y = 0; y < image0.rows; y++)
        {
            for (int x = 0; x < image0.cols; x++)
            {
                image0.at<Vec3b>(y, x)[0] = pix[fff];
                image0.at<Vec3b>(y, x)[1] = pix[fff + 1];
                image0.at<Vec3b>(y, x)[2] = pix[fff + 2];
                fff += 3;
            }
        }
        namedWindow("Image1");
        imshow("Image1", image0);
        
        ///////////////////////////////////////////////////////////
        
         //copy in cuda
        cudaMemcpy(d_ex_pix, pix, rows * cols * 3 * sizeof(uchar), cudaMemcpyHostToDevice);
        memset(pix1, 255, rows * cols * 3);
        cudaMemcpy(d_ex_pix1, pix1, rows * cols * 3 * sizeof(uchar), cudaMemcpyHostToDevice);
        //fish eye
        fish_eye << <rows/2, cols >> > (d_ex_pix, d_ex_pix1, rows, cols);

        cudaMemcpy(pix1, d_ex_pix1, rows * cols * 3 * sizeof(uchar), cudaMemcpyDeviceToHost);
        memcpy(pix_buf1, pix1, rows/2 * cols * 3);
        //byte -> img
        fff = 0;
        for (int y = 0; y < image1.rows; y++)
        {
            for (int x = 0; x < image1.cols; x++)
            {
                image1.at<Vec3b>(y, x)[0] = pix1[fff];
                image1.at<Vec3b>(y, x)[1] = pix1[fff + 1];
                image1.at<Vec3b>(y, x)[2] = pix1[fff + 2];
                fff += 3;
            }
        }
        namedWindow("Image2");
        imshow("Image2", image1);
        
        //////////////////////////////////////////////////////////////////////////////////////
        MPI_Recv(pix_buf, rows/2 * cols * 3, MPI_BYTE, 1, 0, MPI_COMM_WORLD, &status);
        MPI_Send(pix1, rows/2 * cols * 3, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
        
        memcpy(pix1 + rows/2 * cols * 3, pix_buf, rows/2 * cols * 3);
        memcpy(pix, pix1, rows * cols * 3);
        cudaMemcpy(d_ex_pix1, pix1, rows * cols * 3 * sizeof(uchar), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ex_pix, pix, rows * cols * 3 * sizeof(uchar), cudaMemcpyHostToDevice);
        interpol << <rows/2, cols >> > (d_ex_pix, d_ex_pix1, rows, cols);
        cudaMemcpy(pix1, d_ex_pix1, rows * cols * 3 * sizeof(uchar), cudaMemcpyDeviceToHost);
        MPI_Recv(pix_buf, rows/2 * cols * 3, MPI_BYTE, 1, 0, MPI_COMM_WORLD, &status);
        memcpy(pix1 + rows/2 * cols * 3, pix_buf, rows/2 * cols * 3);
    }
    if (rank == 1)
    {
        //alloc memory on CUDA
        cudaMalloc((void**)&d_pix, rows/2 * cols * 3 * sizeof(uchar));
        cudaMalloc((void**)&d_pix1, rows/2 * cols * 3 * sizeof(uchar));
        cudaMalloc((void**)&d_ex_pix, rows * cols * 3 * sizeof(uchar));
        cudaMalloc((void**)&d_ex_pix1, rows * cols * 3 * sizeof(uchar));
        // img -> byte
        for (int y = 0; y < image0.rows; y++)
        {
            for (int x = 0; x < image0.cols; x++)
            {
                pix[fff] = image0.at<Vec3b>(y, x)[0];
                pix[fff + 1] = image0.at<Vec3b>(y, x)[1];
                pix[fff + 2] = image0.at<Vec3b>(y, x)[2];
                fff += 3;
            }
        }
        memset(pix, 0, rows/2 * cols * 3);
        memcpy(pix_buf, pix + rows/2 * cols * 3, rows/2 * cols * 3);
        //copy in cuda
        cudaMemcpy(d_pix, pix_buf, rows/2 * cols * 3 * sizeof(uchar), cudaMemcpyHostToDevice);
        //haff
        haff1 << <rows/2, cols >> > (d_pix,rows,cols);
        //copy in host
        cudaMemcpy(pix_buf, d_pix, rows/2* cols * 3 * sizeof(uchar), cudaMemcpyDeviceToHost);
      
        memcpy(pix + rows/2 * cols * 3, pix_buf, rows/2 * cols * 3);
        //end haff
        fff = 0;
        //byte -> img
        for (int y = 0; y < image0.rows; y++)
        {
            for (int x = 0; x < image0.cols; x++)
            {
                image0.at<Vec3b>(y, x)[0] = pix[fff];
                image0.at<Vec3b>(y, x)[1] = pix[fff + 1];
                image0.at<Vec3b>(y, x)[2] = pix[fff + 2];
                fff += 3;
            }
        }
        namedWindow("Image1");
        imshow("Image1", image0);

        //////////////////////////////////////////////////////////
        
        memcpy(pix + rows / 2 * cols * 3, pix_buf, rows / 2 * cols * 3);
        //copy in cuda
        cudaMemcpy(d_ex_pix, pix, rows * cols * 3 * sizeof(uchar), cudaMemcpyHostToDevice);
        memset(pix1, 255, rows * cols * 3);
        cudaMemcpy(d_ex_pix1, pix1, rows * cols * 3 * sizeof(uchar), cudaMemcpyHostToDevice);
        //fish eye
        fish_eye1 << <rows/2, cols >> > (d_ex_pix, d_ex_pix1, rows, cols);
        //copy in host
        cudaMemcpy(pix1, d_ex_pix1, rows * cols * 3 * sizeof(uchar), cudaMemcpyDeviceToHost);
        memcpy(pix_buf1, pix1 + rows/2 * cols * 3, rows/2 * cols * 3);

        //byte -> img
        fff = 0;
        for (int y = 0; y < image1.rows; y++)
        {
            for (int x = 0; x < image1.cols; x++)
            {
                image1.at<Vec3b>(y, x)[0] = pix1[fff];
                image1.at<Vec3b>(y, x)[1] = pix1[fff + 1];
                image1.at<Vec3b>(y, x)[2] = pix1[fff + 2];
                fff += 3;
            }
        }

        namedWindow("Image2");
        imshow("Image2", image1);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        MPI_Send(pix1 + rows/2 * cols * 3, rows/2 * cols * 3, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(pix_buf, rows/2 * cols * 3, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);

        memcpy(pix1, pix_buf, rows/2 * cols * 3);
        memcpy(pix, pix1, rows * cols * 3);
        cudaMemcpy(d_ex_pix1, pix1, rows * cols * 3 * sizeof(uchar), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ex_pix, pix, rows * cols * 3 * sizeof(uchar), cudaMemcpyHostToDevice);
        interpol1 << <rows/2, cols >> > (d_ex_pix, d_ex_pix1, rows, cols);
        cudaMemcpy(pix1, d_ex_pix1, rows * cols * 3 * sizeof(uchar), cudaMemcpyDeviceToHost);
        MPI_Send(pix1 + rows/2 * cols * 3, rows/2 * cols * 3, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    }
    //end ______________________________________________________________________
    
    fff = 0;
    for (int y = 0; y < image1.rows; y++)
    {
        for (int x = 0; x < image1.cols; x++)
        {
            image2.at<Vec3b>(y, x)[0] = pix1[fff];
            image2.at<Vec3b>(y, x)[1] = pix1[fff + 1];
            image2.at<Vec3b>(y, x)[2] = pix1[fff + 2];
            fff += 3;
        }
    }

    delete pix;
    delete pix_buf;
    delete ex_pix_buf;
    delete pix1;
    delete pix_buf1;
    delete ex_pix_buf1;
    cudaFree(d_pix);
    cudaFree(d_pix1);
    cudaFree(d_ex_pix);
    cudaFree(d_ex_pix1);
    namedWindow("Image3");
    imshow("Image3", image2);
    waitKey();
    MPI_Finalize();
    return 0;
}