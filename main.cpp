#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

// Function to apply the Bayer filter pattern
Mat applyBayerPattern(const Mat& image) {
    Mat bayerImage = Mat::zeros(image.size(), CV_8UC1);

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            Vec3b pixel = image.at<Vec3b>(y, x);
            if ((y % 2 == 0) && (x % 2 == 0)) {
                // G1
                bayerImage.at<uchar>(y, x) = pixel[1];
            }
            else if ((y % 2 == 0) && (x % 2 == 1)) {
                // R2
                bayerImage.at<uchar>(y, x) = pixel[2];
            }
            else if ((y % 2 == 1) && (x % 2 == 0)) {
                // B3
                bayerImage.at<uchar>(y, x) = pixel[0];
            }
            else {
                // G4
                bayerImage.at<uchar>(y, x) = pixel[1];
            }
        }
    }

    return bayerImage;
}



Mat demosaicBayerImage(const Mat& bayerImage) {
    Mat rgbImage = Mat::zeros(bayerImage.size(), CV_8UC3);

    for (int y = 1; y < bayerImage.rows - 1; y++) {
        for (int x = 1; x < bayerImage.cols - 1; x++) {
            Vec3b pixel = Vec3b(0, 0, 0);

            if ((y % 2 == 0) && (x % 2 == 0)) {
                // G1 position (green channel)
                pixel[1] = bayerImage.at<uchar>(y, x);
                pixel[2] = (bayerImage.at<uchar>(y, x - 1) + bayerImage.at<uchar>(y, x + 1)) / 2;
                pixel[0] = (bayerImage.at<uchar>(y - 1, x) + bayerImage.at<uchar>(y + 1, x)) / 2;
            }
            else if ((y % 2 == 0) && (x % 2 == 1)) {
                // R2 position (red channel)
                pixel[2] = bayerImage.at<uchar>(y, x);
                pixel[1] = (bayerImage.at<uchar>(y, x - 1) + bayerImage.at<uchar>(y, x + 1) + bayerImage.at<uchar>(y - 1, x) + bayerImage.at<uchar>(y + 1, x)) / 4;
                pixel[0] = (bayerImage.at<uchar>(y - 1, x - 1) + bayerImage.at<uchar>(y - 1, x + 1) + bayerImage.at<uchar>(y + 1, x - 1) + bayerImage.at<uchar>(y + 1, x + 1)) / 4;
            }
            else if ((y % 2 == 1) && (x % 2 == 0)) {
                // B3 position (blue channel)
                pixel[0] = bayerImage.at<uchar>(y, x);
                pixel[1] = (bayerImage.at<uchar>(y, x - 1) + bayerImage.at<uchar>(y, x + 1) + bayerImage.at<uchar>(y - 1, x) + bayerImage.at<uchar>(y + 1, x)) / 4;
                pixel[2] = (bayerImage.at<uchar>(y - 1, x - 1) + bayerImage.at<uchar>(y - 1, x + 1) + bayerImage.at<uchar>(y + 1, x - 1) + bayerImage.at<uchar>(y + 1, x + 1)) / 4;
            }
            else {
                // G4 position (green channel)
                pixel[1] = bayerImage.at<uchar>(y, x);
                pixel[2] = (bayerImage.at<uchar>(y - 1, x) + bayerImage.at<uchar>(y + 1, x)) / 2;
                pixel[0] = (bayerImage.at<uchar>(y, x - 1) + bayerImage.at<uchar>(y, x + 1)) / 2;
            }

            rgbImage.at<Vec3b>(y, x) = pixel;
        }
    }

    return rgbImage;
}

int main() {
    // Path to the input RGB image
    string inputImagePath = "C:\\Users\\91709\\source\\repos\\OpenCV\\OpenCV\\Utils\\Kodak dataset\\kodim01.png";
    // Path to save the output Bayered image
    string outputImagePath = "C:\\Users\\91709\\source\\repos\\OpenCV\\OpenCV\\Utils\\Bayered_kodim01.png";

    // Read the input image
    Mat image = imread(inputImagePath);
    if (image.empty()) {
        cerr << "Could not open or find the image: " << inputImagePath << endl;
        return -1;
    }

    // Apply the Bayer filter pattern
    Mat bayerImage = applyBayerPattern(image);

    // Save the Bayered image
    if (imwrite(outputImagePath, bayerImage)) {
        cout << "Bayered image saved successfully to: " << outputImagePath << endl;
    }
    else {
        cerr << "Failed to save the Bayered image." << endl;
        return -1;
    }

    // Path to the input Bayer CFA image
    string BayerImagePath = "C:\\Users\\91709\\source\\repos\\OpenCV\\OpenCV\\Utils\\Bayered_kodim01.png";
    // Path to save the output demosaiced RGB image
    string DemosaickedImagePath = "C:\\Users\\91709\\source\\repos\\OpenCV\\OpenCV\\Utils\\Demosaiced_kodim01.png";

    // Read the input Bayer CFA image
    Mat BayerImage_read = imread(BayerImagePath, IMREAD_GRAYSCALE);
    if (BayerImage_read.empty()) {
        cerr << "Could not open or find the image: " << BayerImagePath << endl;
        return -1;
    }

    // Demosaic the Bayer CFA image
    Mat Demosaicked_rgb_Image = demosaicBayerImage(BayerImage_read);

    // Save the demosaiced RGB image
    if (imwrite(DemosaickedImagePath, Demosaicked_rgb_Image)) {
        cout << "Demosaiced image saved successfully to: " << DemosaickedImagePath << endl;
    }
    else {
        cerr << "Failed to save the demosaiced image." << endl;
        return -1;
    }

    string outputYUVImagePath = "C:\\Users\\91709\\source\\repos\\OpenCV\\OpenCV\\Utils\\YUV_kodim01.png";
    Mat yuvImage;
    cvtColor(Demosaicked_rgb_Image, yuvImage, COLOR_BGR2YUV);

    // Save the YUV image
    if (imwrite(outputYUVImagePath, yuvImage)) {
        cout << "YUV image saved successfully to: " << outputYUVImagePath << endl;
    }
    else {
        cerr << "Failed to save the YUV image." << endl;
        return -1;
    }


    vector<Mat> yuvChannels;
    split(yuvImage, yuvChannels);
    Mat uComponent = yuvChannels[1];
    Mat vComponent = yuvChannels[2];

    cout << "Initial U, V dimensions: " << uComponent.rows << " " << vComponent.cols << endl;

    // Downsample U and V components by average pooling with a 2x2 block without overlapping
    Mat u_int_d, v_int_d;
    resize(uComponent, u_int_d, Size(), 0.5, 0.5, INTER_AREA);
    resize(vComponent, v_int_d, Size(), 0.5, 0.5, INTER_AREA);

    cout << "Initial Downsampled U, V dimensions: " << u_int_d.rows << " " << v_int_d.cols << endl;


    //Zero Padding
    int top = 1, bottom = 1, left = 1, right = 1;
    Mat u_padded, v_padded;
    copyMakeBorder(u_int_d, u_padded, top, bottom, left, right, BORDER_CONSTANT, Scalar(0));
    copyMakeBorder(v_int_d, v_padded, top, bottom, left, right, BORDER_CONSTANT, Scalar(0));

    // Check the new size
    cout << "Padded U dimensions: " << u_padded.rows << "x" << u_padded.cols << endl;
    cout << "Padded V dimensions: " << v_padded.rows << "x" << v_padded.cols << endl;

    Mat k_u1 = (Mat_<float>(3, 3) << 1/16,3/16, 0,3/16,0,0,0,0,0);
    Mat k_u2 = (Mat_<float>(3, 3) << 0, 3/16, 1/16, 3/16, 0, 0, 0, 0, 0);
    Mat k_u3 = (Mat_<float>(3, 3) << 0,0,0,3/16, 0, 0, 1/16, 3/16, 0);
    Mat k_u4 = (Mat_<float>(3, 3) << 0,0, 0,0, 0,3/16, 0,3/16,1/16);

    vector<Mat> kernels_u = { k_u1, k_u2, k_u3, k_u4 };
    vector<float> u_cal, v_cal;
    //for (int i = 0; i < u_padded.rows; ++i) {
    //    for (int j = 0; j < u_padded.cols; ++j) {
    //        for (int I = 0; I < 4; I++)
    //        {
    //            float sum_u = 0.0, sum_v = 0.0;
    //            Mat k = kernels_u[0];
    //            for (int ki = 0; ki < 3; ++ki) {
    //                for (int kj = 0; kj < 3; ++kj) {
    //                    int x = j + kj;
    //                    int y = i + ki;
    //                    sum_u += k.at<float>(ki, kj) * u_padded.at<float>(y, x);
    //                    sum_v += k.at<float>(ki, kj) * v_padded.at<float>(y, x);
    //                }
    //            }
    //            u_cal.push_back(sum_u);
    //            v_cal.push_back(sum_v);
    //        }

    //        // Store the result in the output matrix
    //        //u_convolved.at<float>(i, j) = sum;   
    //         
    //    }
    //}
     cout << "Good_Morning";


    return 0;
}
