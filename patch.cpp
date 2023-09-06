/*
    Mohammad Hossein Bagheri
        *** 2071501 ***
    https://datahacker.rs/feature-matching-methods-comparison-in-opencv/ 
    studied this article (it is in python) as the reference on how to use matching in opencv
*/

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
// *****
// Had some problems with xfeatures2d that got fixed this way
#include "C:\opencv\opencv_contrib-4.x\modules\xfeatures2d\include\opencv2\xfeatures2d\nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
// *****
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

// Function that loads the corrupted image (image_to_complete.jpg)
// Takes the image path + the image name as its argument
Mat loadCorruptedImage(const string& imagePath) {
    
    //returns the image
    return imread(imagePath, IMREAD_GRAYSCALE);

}

// Function that loads both regular and _t patches to work with
// Takes the directory which the patch files are
vector<Mat> loadAllPatches(const string& patchesPath) {
    // the vector of patches
    vector<Mat> patches;
    // i=4 because we had 4 patches in each category
    for (int i = 0; i < 4; ++i) {
        // loads normal patches
        string patchPath = patchesPath + "\\patch_" + to_string(i) + ".jpg";
        Mat patch = imread(patchPath, IMREAD_GRAYSCALE);
        patches.push_back(patch);
        // loads the transformed patches
        string patchPath_t = patchesPath + "\\patch_t_" + to_string(i) + ".jpg";
        Mat patch_t = imread(patchPath_t, IMREAD_GRAYSCALE);
        patches.push_back(patch_t);
    }
    // returns a vector of the load patches
    return patches;
}

// Function that extracts the SIFT features of the image (for both the corrupted and patch images)
// It takes one empty matrix to fill it with the descriptor values and one vector of keypoint to store the keypoints
void extractSiftFeatures(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) {
    Ptr<SIFT> sift = SIFT::create();
    sift->detectAndCompute(image, noArray(), keypoints, descriptors);
}

// Function that computes the match between the image and the given patches
// It needs image and patch features to match them
vector<vector<DMatch>> computeMatch(const Mat& imageFeatures, const Mat& patchFeatures) {
    
    // We use BFMatcher (Brute Force) with NORM_L2
    BFMatcher matcher(NORM_L2);

    vector<vector<DMatch>> matches;
    int k = 2;
    // matches features using KNN algorithm
    matcher.knnMatch(imageFeatures, patchFeatures, matches, k);
    // returns a vector of DMatch vectors
    return matches;
}

// Function that refines the matches to get better results faster (Lowe's ratio test)
vector<DMatch> refineMatch(const vector<vector<DMatch>>& matches, float ratio) {
    vector<DMatch> refinedMatches;
    for (const auto& m : matches)
    {
        int k = 2;
        if (m.size() < k)
        {
            continue;
        }

        const DMatch& bestMatch = m[0];
        const DMatch& secondBestMatch = m[1];

        // It basically tests if the bestMatch is smaller than ratio * secondBestMatch
        float distanceRatio = bestMatch.distance / secondBestMatch.distance;
        if (distanceRatio < ratio)
        {
            refinedMatches.push_back(bestMatch);
        }
    }
    // returns the refined matches
    return refinedMatches;
}

// Function that finds the transformation using findHomography and RANSAC
Mat findTransformation(const vector<KeyPoint>& imageKeypoints, const vector<KeyPoint>& patchKeypoints,
    const vector<DMatch>& matches, vector<char>& inliers) {
    
    vector<Point2f> imagePoints;
    vector<Point2f> patchPoints;

    for (const auto& match : matches)
    {
        imagePoints.push_back(imageKeypoints[match.queryIdx].pt);
        patchPoints.push_back(patchKeypoints[match.trainIdx].pt);
    }
    
    Mat homography = findHomography(patchPoints, imagePoints, RANSAC, 3, inliers);

    // returns the calculated homography
    return homography;
}


// Function that overlays the patches on the corrupted image using the found homography
// It needs corrupted image, patch and their homography
void overlayPatches(Mat& image, const Mat& patch, const Mat& homography) {
    Mat transformedPatch;
    warpPerspective(patch, transformedPatch, homography, image.size());
    transformedPatch.copyTo(image, transformedPatch > 0);
}

int main(int argc, char** argv)
{
    // Loads and shows the corrupted image
    Mat image = loadCorruptedImage("D:\\University\\Second Semester\\Computer Vision\\16_lab4\\starwars\\image_to_complete.jpg");
    imshow("Corrupted Image", image);

    // Loads all the patches (patch_n and patch_t_n)
    vector<Mat> patches = loadAllPatches("D:\\University\\Second Semester\\Computer Vision\\16_lab4\\starwars");

    // Extract SIFT features from the corrupted image
    vector<KeyPoint> imageKeypoints;
    Mat imageFeatures;
    extractSiftFeatures(image, imageKeypoints, imageFeatures);

    // Extract SIFT features from all the patches
    vector<vector<KeyPoint>> patchKeypoints(patches.size());
    vector<Mat> patchFeatures(patches.size());
    for (int i = 0; i < patches.size(); ++i) {
        extractSiftFeatures(patches[i], patchKeypoints[i], patchFeatures[i]);
    }

    // For each patch, compute matches and refine them
    for (int i = 0; i < patches.size(); ++i) {
        vector<vector<DMatch>> matches = computeMatch(imageFeatures, patchFeatures[i]);
        vector<DMatch> refinedMatches = refineMatch(matches, 0.6);

        // Finds the transformation using RANSAC
        vector<char> inliers;
        Mat homography = findTransformation(imageKeypoints, patchKeypoints[i], refinedMatches, inliers);

        // Overlays each patch on the image
        overlayPatches(image, patches[i], homography);

    }

    // Display the fixed image
    imshow("Fixed Image", image);
    
    waitKey();
    return EXIT_SUCCESS;
}