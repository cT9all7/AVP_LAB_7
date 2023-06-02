#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <stdlib.h>
#define _USE_MATH_DEFINES

using namespace cv;
using namespace std;
int main(int argc, char* argv[])
{
    Mat image0 = imread("./222.png");
    if (image0.empty())
    {
        std::cerr << "Could not load image";
        return 1;
    }
    cout << image0.rows << "x" << image0.cols << endl;// <---строки(rows) и столбцы(cols) соответственно (y,x)
    //рисуем окружность
    Point center(200, 150);
    int r = 40;
    Scalar LineColor(0, 0, 255);
    int thickness = 1;
    namedWindow("Image0");
    circle(image0, center, r, LineColor, thickness);
    imshow("Image0", image0);
    //waitKey();
    //тут хаф, поиск окружности;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

    for (int y = 0; y < image0.rows; y++)
    {
        for (int x = 0; x < image0.cols; x++)
        {
            if (pow((200 - x), 2) + pow((150 - y), 2) <= pow(40, 2) && pow((200 - x), 2) + pow((150 - y), 2) > pow(39, 2))
           // if (pow((200 - x), 2) + pow((150 - y), 2) == pow(40, 2))
            {
               image0.at<Vec3b>(y, x)[0] = 0;// blue
               image0.at<Vec3b>(y, x)[1] = 255;// green
               image0.at<Vec3b>(y, x)[2] = 255;// red
               // cout << "x = " << j << " y = " << i << endl;
            }
        }
    }
    
    namedWindow("Image1");
    imshow("Image1", image0);
    //waitKey();

    //вернули что было у хафа;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

    for (int y = 0; y < image0.rows; y++)
    {
        for (int x = 0; x < image0.cols; x++)
        {
            if (pow((200 - x), 2) + pow((150 - y), 2) <= pow(40, 2) && pow((200 - x), 2) + pow((150 - y), 2) > pow(39, 2))
            {
                image0.at<Vec3b>(y, x)[0] = 0;// blue
                image0.at<Vec3b>(y, x)[1] = 0;// green
                image0.at<Vec3b>(y, x)[2] = 255;// red
                // cout << "x = " << j << " y = " << i << endl;
            }
        }
    }
    namedWindow("Image2");
    imshow("Image2", image0);

    //рыбий глаз ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

    Mat image1 = imread("./1.png"); //поле для копирования
    float N = (float)image1.cols; //столбцы
    float M = (float)image1.rows; //строки
    float omega = 0.8; //параметр для "рыбьего глаза" с ним играться будем
    float x0 = 0, y0 = 0, r0 = 0, teta = 0, scale = 0, r1 = 0;
    int x1, y1;
    float e = 0.01;
    Vec3b pix;
    
    cout << endl << e << endl;

    for (float y = 0; y < image0.rows; y++)
    {
        for (float x = 0; x < image0.cols; x++)
        {
            x0 = (x - ((N + 1) / 2)) * 2 / N;

            y0 = (y - ((M + 1) / 2)) * 2 / M;

            r0 = sqrt(pow(x0, 2) + pow(y0, 2));

            teta = atan2(y0, x0);

            scale = min( 1 / (abs(cos(teta) + e)), 1 / (abs(sin(teta) + e)) );

            r1 = min(scale, (float)1.0) * pow(r0, omega);
            

            x1 = ((N / 2) * r1 * cos(teta) + ((N + 1) / 2));
            
            y1 = ((M / 2) * r1 * sin(teta) + ((M + 1) / 2));
            pix[0] = image0.at<Vec3b>(y, x)[0];
            pix[1] = image0.at<Vec3b>(y, x)[1];
            pix[2] = image0.at<Vec3b>(y, x)[2];
            image1.at<Vec3b>(y1, x1)[0] = pix[0];
            image1.at<Vec3b>(y1, x1)[1] = pix[1];
            image1.at<Vec3b>(y1, x1)[2] = pix[2];
            
        }
    }
    namedWindow("Image3");
    imshow("Image3", image1);
    waitKey();

    float alpha_x, alpha_y;
    float tx0, tx1, tx, ty0, ty1, ty;

    float alpha;
    float t, t0, t1;

    for (float y = 0; y < image1.rows; y++)
    {
        for (float x = 0; x < image1.cols; x++)
        {   
            if (image1.at<Vec3b>(y, x)[0] == 255 && image1.at<Vec3b>(y, x)[1] == 255 && image1.at<Vec3b>(y, x)[2] == 255)
            {
                for (int j = 1; y + j < image1.rows; j++)
                {
                    if (image1.at<Vec3b>(y + j, x)[0] != 255 && image1.at<Vec3b>(y + j, x)[1] != 255 && image1.at<Vec3b>(y + j, x)[2] != 255)
                    {
                        ty0 = y + j;
                        break;
                    }
                }
                for (int j = -1; y + j == 0; j--)
                {
                    if (image1.at<Vec3b>(y + j, x)[0] != 255 && image1.at<Vec3b>(y + j, x)[1] != 255 && image1.at<Vec3b>(y + j, x)[2] != 255)
                    {
                        ty1 = y - j;
                        break;
                    }
                }
                ty = y;
                alpha_y = (ty1 - ty) / (ty1 - ty0);

                if (x == 200)
                {
                    image1.at<Vec3b>(y, x)[0] = image1.at<Vec3b>(y, x-1)[0];
                    image1.at<Vec3b>(y, x)[1] = image1.at<Vec3b>(y, x-1)[1];
                    image1.at<Vec3b>(y, x)[2] = image1.at<Vec3b>(y, x-1)[2];
                }
               
                //answer
                
                 else
                 {
                     image1.at<Vec3b>(y, x)[0] = image1.at<Vec3b>(ty0, x)[0] * alpha_y + (1 - alpha_y) * image1.at<Vec3b>(ty1, x)[0];
                     image1.at<Vec3b>(y, x)[1] = image1.at<Vec3b>(ty0, x)[1] * alpha_y + (1 - alpha_y) * image1.at<Vec3b>(ty1, x)[1];
                    image1.at<Vec3b>(y, x)[2] = image1.at<Vec3b>(ty0, x)[2] * alpha_y + (1 - alpha_y) * image1.at<Vec3b>(ty1, x)[2];
                 }
            }

        }
    }
    namedWindow("Image4");
    imshow("Image4", image1);
    waitKey();
    return 0;

}
