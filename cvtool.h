/**
MIT License
Copyright (c) 2020 bbqz007 <https://github.com/bbqz007, http://www.cnblogs.com/bbqzsl>
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef ZCVTOOL_HELPER__H_
#define ZCVTOOL_HELPER__H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>

//#include <opencv2/imgcodecs.hpp>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
using namespace cv;
using namespace std;

namespace zhelper
{

namespace cvtool
{

class itf_filter;
itf_filter* createFilter(const char* filter, const string& name);

class filter_graph;
class itf_filter
{
public:
    itf_filter(const string& name) : num_(++snum_)
    {
        ostringstream os;
        os << num_ << ": " << name << " -=>@github.com/bbqz007";
        name_ = os.str();
        namedWindow(name_);
    }
    virtual ~itf_filter()
    {
        destroyWindow(name_);
    }
    Mat filter(Mat& image)
    {
        return _filter(image);
    }
protected:
    static void update_(int pos, void* userdata);
    void update_next_(Mat image);
    virtual Mat _filter(Mat& image) = 0;
    string name_;
    const int num_;
    static int snum_;
    filter_graph* graph_;
    friend class filter_graph;
};

typedef shared_ptr<itf_filter> sptr_filter;

class filter_graph
{
public:
    Mat filter(Mat& image)
    {
        tmp_ = image;
        return filter();
    }
    void open(const string& cmd);
    void push(sptr_filter& filter)
    {
        if (filter)
        {
            filter->graph_ = this;
            filters_.push_back(filter);
        }

    }
    filter_graph& apply(function<void(Mat)> f)
    {
        apply_ = f;
        return *this;
    }
    Mat origin() { return (retmp_.empty())?tmp_:retmp_; }
    void update_origin(Mat m) { retmp_ = m; }
protected:
    Mat filter()
    {
        Mat res = tmp_;
        for_each(filters_.begin(), filters_.end(),
                 [&](sptr_filter& filter){
                    res = filter->filter(res);
                 });
        if (apply_)
            apply_(res);
        return res;
    }
    void filter_next(itf_filter* f, Mat image)
    {
        Mat res;
        auto it = filters_.begin();
        for(int i = 0; i < filters_.size(); ++i, ++it)
        {
            if (filters_[i].get() == f)
            {
                break;
            }
        }
        if (it != filters_.end())
        {
            res = image;
            for_each(++it, filters_.end(),
                 [&](sptr_filter& filter){
                    res = filter->filter(res);
                 });
            if (apply_)
                apply_(res);
        }
    }
    friend class itf_filter;
    vector<sptr_filter> filters_;
    Mat tmp_;
    Mat retmp_;
    function<void(Mat)> apply_;
};

void itf_filter::update_(int pos, void* userdata)
{
    itf_filter* f = (itf_filter*)userdata;
    if (f)
        f->graph_->filter();
}

void itf_filter::update_next_(Mat image)
{
    graph_->filter_next(this, image);
}

int itf_filter::snum_ = 0;

class noop_filter : public itf_filter
{
public:
    noop_filter(const string& name) : itf_filter(name)
    {
        noop_ = 128;
        createTrackbar("noop", name_, &noop_, 256, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        (void)image;
        return res;
    }
    int noop_;
};

class threshold_filter : public itf_filter
{
public:
    threshold_filter(const string& name) : itf_filter(name)
    {
        threshval_ = 128;
        createTrackbar("threshval", name_, &threshval_, 255, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat bw = threshval_ < 128 ? (image < threshval_) : (image > threshval_);
        imshow(name_, bw);
        return bw;
    }
    int threshval_;
};

class Canny_filter : public itf_filter
{
public:
    Canny_filter(const string& name) : itf_filter(name)
    {
        threshval1_ = 0;
        threshval2_ = 50;
        aperturesize_ = 5;
        l2gradient_ = 0;
        createTrackbar("threshval-1", name_, &threshval1_, 255, itf_filter::update_, this);
        createTrackbar("threshval-2", name_, &threshval2_, 255, itf_filter::update_, this);
        createTrackbar("aperture size |1", name_, &aperturesize_, 7, itf_filter::update_, this);
        createTrackbar("L2 gradient (OFF/ON)", name_, &l2gradient_, 1, itf_filter::update_, this);
        setTrackbarMin("aperture size |1", name_, 3);
        setTrackbarMax("aperture size |1", name_, 7);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        Canny(image, res, threshval1_, threshval2_, aperturesize_|1, l2gradient_);
        imshow(name_, res);
        return res;
    }
    int threshval1_;
    int threshval2_;
    int aperturesize_;
    int l2gradient_;
};

class HoughLines_filter : public itf_filter
{
public:
    HoughLines_filter(const string& name) : itf_filter(name)
    {
        rho_ = 50;
        theta_ = 0;
        threshval_ = 0;
        createTrackbar("rho*.1", name_, &rho_, 1000, itf_filter::update_, this);
        createTrackbar("theta*.1", name_, &theta_, 1000, itf_filter::update_, this);
        createTrackbar("threshval", name_, &threshval_, 7, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res(image.size(), CV_8UC1);
        HoughLines(image, res, rho_*.1, theta_*.1, threshval_);
        imshow(name_, res);
        return res;
    }
    int rho_;
    int theta_;
    int threshval_;
};

class morphology_filter : public itf_filter
{
public:
    morphology_filter(const string& name) : itf_filter(name)
    {
        method_ = 0;
        threshval_ = 0;
        shape_ = 0;
        kernel_ = 1;
        createTrackbar("morphology", name_, &threshval_, 20, itf_filter::update_, this);
        setTrackbarMin("morphology", name_, -10);
        setTrackbarMax("morphology", name_, 10);
        //setTrackbarPos("morphology", name_, 0);
        createTrackbar("method:\n0-open\n1-close\n2-erode\n3-dilate\n4-gradient\n5-tophat\n6-blackhat", name_, &method_, 6, itf_filter::update_, this);
        createTrackbar("rect/ellipse/cross", name_, &shape_, 2, itf_filter::update_, this);
        createTrackbar("kernel(OFF/ON)", name_, &kernel_, 1, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        int n = threshval_;
        int an = abs(n);
        Mat element;
        if (kernel_)
        switch (shape_)
        {
        case 0:
            element = getStructuringElement(MORPH_RECT, Size(an*2+1, an*2+1), Point(an, an) );
            break;
        case 1:
            element = getStructuringElement(MORPH_ELLIPSE, Size(an*2+1, an*2+1), Point(an, an) );
            break;
        case 2:
            element = getStructuringElement(MORPH_CROSS, Size(an*2+1, an*2+1), Point(an, an) );
            break;
        }
        switch (method_)
        {
        case 0:
            morphologyEx(image, res, MORPH_OPEN, element);
            break;
        case 1:
            morphologyEx(image, res, MORPH_CLOSE, element);
            break;
        case 2:
            erode(image, res, element);
            break;
        case 3:
            dilate(image, res, element);
            break;
        case 4:
            morphologyEx(image, res, MORPH_GRADIENT, element);
            break;
        case 5:
            morphologyEx(image, res, MORPH_TOPHAT, element);
            break;
        case 6:
            morphologyEx(image, res, MORPH_BLACKHAT, element);
            break;
        //case 7: // src.type() == CV_8UC1
         //   morphologyEx(image, res, MORPH_HITMISS, element);
          //  break;
        }
        imshow(name_, res);
        return res;
    }
    int threshval_;
    int method_;
    int shape_;
    int kernel_;

};

class medianBlur_filter : public itf_filter
{
public:
    medianBlur_filter(const string& name) : itf_filter(name)
    {
        ksize_ = 1;
        createTrackbar("ksize|1", name_, &ksize_, 11, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        medianBlur(image, res, ksize_|1);
        imshow(name_, res);
        return res;
    }
    int ksize_;
};

class GaussianBlur_filter : public itf_filter
{
public:
    GaussianBlur_filter(const string& name) : itf_filter(name)
    {
        ksize_ = 1;
        sigma_ = 1;
        createTrackbar("ksize|1", name_, &ksize_, 11, itf_filter::update_, this);
        createTrackbar("sigma*.5", name_, &sigma_, 15, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        GaussianBlur(image, res, Size(ksize_|1, ksize_|1), sigma_*.5, sigma_*.5);
        imshow(name_, res);
        return res;
    }
    int ksize_;
    int sigma_;
};

class bilateral_filter : public itf_filter
{
public:
    bilateral_filter(const string& name) : itf_filter(name)
    {
        d_ = 5;
        sigmaC_ = 100;
        sigmaS_ = 100;
        createTrackbar("d", name_, &d_, 11, itf_filter::update_, this);
        createTrackbar("sigmaColor", name_, &sigmaC_, 300, itf_filter::update_, this);
        createTrackbar("sigmaSpace", name_, &sigmaS_, 300, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        bilateralFilter(image, res, d_, sigmaC_, sigmaS_);
        imshow(name_, res);
        return res;
    }
    int d_;
    int sigmaC_;
    int sigmaS_;
};

class box_filter : public itf_filter
{
public:
    box_filter(const string& name) : itf_filter(name)
    {
        ddepth_ = 0;
        ksize_ = 3;
        createTrackbar("ddepth(-1,8U,8S,16U,16S,32S,32F)", name_, &ddepth_, 7, itf_filter::update_, this);
        createTrackbar("ksize|1", name_, &ksize_, 11, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        boxFilter(image, res, ddepth_-1, Size(ksize_|1, ksize_|1));
        imshow(name_, res);
        return res;
    }
    int ddepth_;
    int ksize_;
};

class sqrBox_filter : public itf_filter
{
public:
    sqrBox_filter(const string& name) : itf_filter(name)
    {
        ddepth_ = 0;
        ksize_ = 3;
        createTrackbar("ddepth(-1,8U,8S,16U,16S,32S,32F)", name_, &ddepth_, 7, itf_filter::update_, this);
        createTrackbar("ksize|1", name_, &ksize_, 11, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        sqrBoxFilter(image, res, ddepth_-1, Size(ksize_|1, ksize_|1));
        imshow(name_, res);
        return res;
    }
    int ddepth_;
    int ksize_;
};

class blur_filter : public itf_filter
{
public:
    blur_filter(const string& name) : itf_filter(name)
    {
        ksize_ = 3;
        createTrackbar("ksize|1", name_, &ksize_, 11, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        blur(image, res, Size(ksize_|1, ksize_|1));
        imshow(name_, res);
        return res;
    }
    int ksize_;
};

class filter2D_filter;
class sepFilter2D_filter;

class Sobel_filter : public itf_filter
{
public:
    Sobel_filter(const string& name) : itf_filter(name)
    {
        dx_ = dy_ = 1;
        ddepth_ = 0;
        ksize_ = 3;
        createTrackbar("dx", name_, &dx_, 10, itf_filter::update_, this);
        setTrackbarMax("dx", name_, max(((ksize_|1) - 1), 1));
        createTrackbar("dy", name_, &dy_, 10, itf_filter::update_, this);
        setTrackbarMax("dy", name_, max(((ksize_|1) - 1), 1));
        createTrackbar("ddepth(-1,8U,8S,16U,16S,32S,32F)", name_, &ddepth_, 7, itf_filter::update_, this);
        createTrackbar("ksize|1", name_, &ksize_, 11, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        setTrackbarMax("dx", name_, max(((ksize_|1) - 1), 1));
        setTrackbarMax("dy", name_, max(((ksize_|1) - 1), 1));
        if (dx_ == 0)
            setTrackbarMin("dy", name_, 1);
        if (dy_ == 0)
            setTrackbarMin("dx", name_, 1);
        if (dx_ && dy_)
        {
            setTrackbarMin("dy", name_, 0);
            setTrackbarMin("dx", name_, 0);
        }
        Sobel(image, res, ddepth_-1, dx_, dy_, ksize_|1);
        imshow(name_, res);
        return res;
    }
    int dx_;
    int dy_;
    int ddepth_;
    int ksize_;
};

class Scharr_filter : public itf_filter
{
public:
    Scharr_filter(const string& name) : itf_filter(name)
    {
        dx_ = dy_ = 0;
        ddepth_ = 0;
        createTrackbar("dx(0,1)", name_, &dx_, 1, itf_filter::update_, this);
        //createTrackbar("dy*.01", name_, &dy_, 1, itf_filter::update_, this);
        createTrackbar("ddepth(-1,8U,8S,16U,16S,32S,32F)", name_, &ddepth_, 7, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        Scharr(image, res, ddepth_-1, dx_, 1 - dx_);
        imshow(name_, res);
        return res;
    }
    int dx_;
    int dy_;
    int ddepth_;
};

class Laplacian_filter : public itf_filter
{
public:
    Laplacian_filter(const string& name) : itf_filter(name)
    {
        ddepth_ = 0;
        ksize_ = 3;
        delta_ = 0;
        createTrackbar("ddepth(-1,8U,8S,16U,16S,32S,32F)", name_, &ddepth_, 7, itf_filter::update_, this);
        createTrackbar("ksize|1", name_, &ksize_, 11, itf_filter::update_, this);
        createTrackbar("delta", name_, &delta_, 11, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        Laplacian(image, res, (ddepth_)?ddepth_*8:-1, ksize_|1, 1, delta_);
        imshow(name_, res);
        return res;
    }
    int ddepth_;
    int ksize_;
    int delta_;
};

class cornerMinEigenVal_filter : public itf_filter
{
public:
    cornerMinEigenVal_filter(const string& name) : itf_filter(name)
    {
        blocksize_ = 3;
        ksize_ = 7;
        createTrackbar("block size", name_, &blocksize_, 7, itf_filter::update_, this);
        setTrackbarMin("block size", name_, 1);
        createTrackbar("ksize|1", name_, &ksize_, 11, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        Mat src;
        if (!(image.type() == CV_8UC1 || image.type() == CV_32FC1))
        {
            cvtColor(image, src, CV_BGR2GRAY);
        }
        else
        {
            src = image;
        }
        cornerMinEigenVal(src, res, blocksize_, ksize_|1);
        imshow(name_, res);
        return res;
    }
    int blocksize_;
    int ksize_;
};

class cornerHarris_filter : public itf_filter
{
public:
    cornerHarris_filter(const string& name) : itf_filter(name)
    {
        blocksize_ = 3;
        ksize_ = 7;
        k_ = 11;
        createTrackbar("block size", name_, &blocksize_, 7, itf_filter::update_, this);
        setTrackbarMin("block size", name_, 1);
        createTrackbar("ksize|1", name_, &ksize_, 11, itf_filter::update_, this);
        createTrackbar("k*.25", name_, &ksize_, 31, itf_filter::update_, this);

    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        Mat src;
        if (!(image.type() == CV_8UC1 || image.type() == CV_32FC1))
        {
            cvtColor(image, src, CV_BGR2GRAY);
        }
        else
        {
            src = image;
        }
        cornerHarris(src, res, blocksize_, ksize_|1, k_*.25);
        imshow(name_, res);
        return res;
    }
    int blocksize_;
    int ksize_;
    int k_;
};

class cornerEigenValsAndVecs_filter : public itf_filter
{
public:
    cornerEigenValsAndVecs_filter(const string& name) : itf_filter(name)
    {
        blocksize_ = 3;
        ksize_ = 7;
        createTrackbar("block size", name_, &blocksize_, 7, itf_filter::update_, this);
        setTrackbarMin("block size", name_, 1);
        createTrackbar("ksize|1", name_, &ksize_, 11, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        Mat src;
        if (!(image.type() == CV_8UC1 || image.type() == CV_32FC1))
        {
            cvtColor(image, src, CV_BGR2GRAY);
        }
        else
        {
            src = image;
        }
        cornerEigenValsAndVecs(src, res, blocksize_, ksize_|1);
        imshow(name_, res);
        return res;
    }
    int blocksize_;
    int ksize_;
};

class preCornerDetect_filter : public itf_filter
{
public:
    preCornerDetect_filter(const string& name) : itf_filter(name)
    {
        ksize_ = 3;
        createTrackbar("ksize|1", name_, &ksize_, 11, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        preCornerDetect(image, res, ksize_|1);
        imshow(name_, res);
        return res;
    }
    int ksize_;
};

class warpAffine_filter;
class warpAffine_filter : public itf_filter
{
public:
    warpAffine_filter(const string& name) : itf_filter(name)
    {
        motion_ = 2;
        size_ = 200;
        createTrackbar("TRANSLATION/EUCLIDEAN/AFFINE", name_, &motion_, 2, itf_filter::update_, this);
        createTrackbar("size", name_, &size_, 8192, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        setTrackbarMax("size", name_, image.cols);
        Mat res;
        Mat warpGround;
        RNG rng(getTickCount());
        double angle;
        switch (motion_)
        {
        case 0: //MOTION_TRANSLATION:
            warpGround = (Mat_<float>(2,3) << 1, 0, (rng.uniform(10.f, 20.f)),
                0, 1, (rng.uniform(10.f, 20.f)));
            //warpAffine(image, res, warpGround,
            //    Size(200,200), INTER_LINEAR + WARP_INVERSE_MAP);
            break;
        case 1: //MOTION_EUCLIDEAN:
            angle = CV_PI/30 + CV_PI*rng.uniform((double)-2.f, (double)2.f)/180;

            warpGround = (Mat_<float>(2,3) << cos(angle), -sin(angle), (rng.uniform(10.f, 20.f)),
                sin(angle), cos(angle), (rng.uniform(10.f, 20.f)));
            //warpAffine(image, res, warpGround,
            //    Size(200,200), INTER_LINEAR + WARP_INVERSE_MAP);
            break;
        case 2: //MOTION_AFFINE:

            warpGround = (Mat_<float>(2,3) << (1-rng.uniform(-0.05f, 0.05f)),
                (rng.uniform(-0.03f, 0.03f)), (rng.uniform(10.f, 20.f)),
                (rng.uniform(-0.03f, 0.03f)), (1-rng.uniform(-0.05f, 0.05f)),
                (rng.uniform(10.f, 20.f)));
            //warpAffine(image, res, warpGround,
            //    Size(200,200), INTER_LINEAR + WARP_INVERSE_MAP);
            break;
        case 3: //MOTION_HOMOGRAPHY:
            warpGround = (Mat_<float>(3,3) << (1-rng.uniform(-0.05f, 0.05f)),
                (rng.uniform(-0.03f, 0.03f)), (rng.uniform(10.f, 20.f)),
                (rng.uniform(-0.03f, 0.03f)), (1-rng.uniform(-0.05f, 0.05f)),(rng.uniform(10.f, 20.f)),
                (rng.uniform(0.0001f, 0.0003f)), (rng.uniform(0.0001f, 0.0003f)), 1.f);
            //warpPerspective(image, res, warpGround,
            //    Size(200,200), INTER_LINEAR + WARP_INVERSE_MAP);
            break;
        }
        warpAffine(image, res, warpGround, Size(size_, size_), INTER_LINEAR + WARP_INVERSE_MAP);
        imshow(name_, res);
        return res;
    }
    int motion_;
    int size_;
};
class warpPerspective_filter;
class warpPerspective_filter : public itf_filter
{
public:
    warpPerspective_filter(const string& name) : itf_filter(name)
    {
        motion_ = 3;
        size_ = 200;
        createTrackbar("size", name_, &size_, 8192, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        setTrackbarMax("size", name_, image.cols);
        Mat res;
        Mat warpGround;
        RNG rng(getTickCount());
        double angle;
        switch (motion_)
        {
        case 3: //MOTION_HOMOGRAPHY:
        default:
            warpGround = (Mat_<float>(3,3) << (1-rng.uniform(-0.05f, 0.05f)),
                (rng.uniform(-0.03f, 0.03f)), (rng.uniform(10.f, 20.f)),
                (rng.uniform(-0.03f, 0.03f)), (1-rng.uniform(-0.05f, 0.05f)),(rng.uniform(10.f, 20.f)),
                (rng.uniform(0.0001f, 0.0003f)), (rng.uniform(0.0001f, 0.0003f)), 1.f);
            //warpPerspective(image, res, warpGround,
            //    Size(200,200), INTER_LINEAR + WARP_INVERSE_MAP);
            break;
        }
        warpPerspective(image, res, warpGround, Size(size_, size_), INTER_LINEAR + WARP_INVERSE_MAP);
        imshow(name_, res);
        return res;
    }
    int motion_;
    int size_;
};

class remap_filter;

class warpPolar_filter;
class warpPolar_filter : public itf_filter
{
public:
    warpPolar_filter(const string& name) : itf_filter(name)
    {
        x_ = 50;
        y_ = 50;
        radius_ = 70;
        flag_ = 0;
        inv_ = 0;
        createTrackbar("x (0~100%)", name_, &x_, 100, itf_filter::update_, this);
        createTrackbar("y (0~100%)", name_, &y_, 100, itf_filter::update_, this);
        createTrackbar("radius (0~100%)", name_, &radius_, 100, itf_filter::update_, this);
        createTrackbar("LINEAR/LOG", name_, &flag_, 1, itf_filter::update_, this);
        createTrackbar("INVERSE (OFF/ON)", name_, &inv_, 1, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        Point2f pt( (float)image.cols * x_ / 100., (float)image.rows * y_ / 100. );
        double maxRadius = radius_ * min(pt.y, pt.x) / 100.;
        int flags = ((flag_)? WARP_POLAR_LOG:WARP_POLAR_LINEAR)
                    + ((inv_)?WARP_INVERSE_MAP:0);

        warpPolar(image, res, Size(),pt, maxRadius, flags);

        imshow(name_, res);
        return res;
    }
    int x_;
    int y_;
    int radius_;
    int flag_;
    int inv_;
};

class pyrDown_filter : public itf_filter
{
public:
    pyrDown_filter(const string& name) : itf_filter(name)
    {
        x_ = 50;
        y_ = 50;
        createTrackbar("x (0~100%)", name_, &x_, 100, itf_filter::update_, this);
        createTrackbar("y (0~100%)", name_, &y_, 100, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        pyrDown(image, res, Size(image.cols * x_ / 100, (float)image.rows *x_ / 100.));
        imshow(name_, res);
        return res;
    }
    int x_;
    int y_;
};

class pyrUp_filter : public itf_filter
{
public:
    pyrUp_filter(const string& name) : itf_filter(name)
    {
        x_ = 200;
        y_ = 200;
        createTrackbar("x (0~100%)", name_, &x_, 200, itf_filter::update_, this);
        createTrackbar("y (0~100%)", name_, &y_, 200, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        pyrUp(image, res, Size(image.cols * x_ / 100, (float)image.rows * x_ / 100.));
        imshow(name_, res);
        return res;
    }
    int x_;
    int y_;
};

// demonstrates
class dem_filter : public itf_filter
{
public:
    dem_filter(const string& name) : itf_filter(name)
    {
        brightness_ = 100;
        contrast_ = 100;
        createTrackbar("brightness-100", name_, &brightness_, 200, itf_filter::update_, this);
        createTrackbar("contrast-100", name_, &contrast_, 200, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        int brightness = brightness_ - 100;
        int contrast = contrast_ - 100;
        /*
         * The algorithm is by Werner D. Streidt
         * (http://visca.com/ffactory/archives/5-99/msg00021.html)
         */
        double a, b;
        if( contrast > 0 )
        {
            double delta = 127.*contrast/100;
            a = 255./(255. - delta*2);
            b = a*(brightness - delta);
        }
        else
        {
            double delta = -128.*contrast/100;
            a = (256.-delta*2)/255.;
            b = a*brightness + delta;
        }
        image.convertTo(res, CV_8U, a, b);
        imshow(name_, res);
        return res;
    }
    int brightness_;
    int contrast_;
};

class distrans_filter : public itf_filter
{
public:
    distrans_filter(const string& name) : itf_filter(name)
    {
        maskSize0_ = DIST_MASK_5;
        voronoiType_ = -1;
        edgeThresh_ = 100;
        distType0_ = DIST_L1;
        method_ = 0;
        createTrackbar("Brightness Threshold", name_, &edgeThresh_, 255, itf_filter::update_, this);
        createTrackbar("method:\n"
            "0 - use C/Inf metric\n"
            "1 - use L1 metric\n"
            "2 - use L2 metric\n"
            "3 - use 3x3 mask\n"
            "4 - use 5x5 mask\n"
            "5 - use precise distance transform\n"
            "6 - switch to Voronoi diagram mode\n"
            "7 - switch to pixel-based Voronoi diagram mode\n", name_,
            &method_, 7, itf_filter::update_, this);

    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat gray = image;
        Mat res;
        if (method_ < 6)
            voronoiType_ = -1;
        switch (method_)
        {
        case 0: distType0_ = DIST_C; break;
        case 1: distType0_ = DIST_L1; break;
        case 2: distType0_ = DIST_L2; break;
        case 3: maskSize0_ = DIST_MASK_3; break;
        case 4: maskSize0_ = DIST_MASK_5; break;
        case 5: maskSize0_ = DIST_MASK_PRECISE; break;
        case 6: voronoiType_ = 0; break;
        case 7: voronoiType_ = 1; break;
        }

        static const Scalar colors[] =
        {
            Scalar(0,0,0),
            Scalar(255,0,0),
            Scalar(255,128,0),
            Scalar(255,255,0),
            Scalar(0,255,0),
            Scalar(0,128,255),
            Scalar(0,255,255),
            Scalar(0,0,255),
            Scalar(255,0,255)
        };

        int maskSize = voronoiType_ >= 0 ? DIST_MASK_5 : maskSize0_;
        int distType = voronoiType_ >= 0 ? DIST_L2 : distType0_;

        Mat edge = gray >= edgeThresh_, dist, labels, dist8u;

        if( voronoiType_ < 0 )
            distanceTransform(edge, dist, distType, maskSize);
        else
            distanceTransform(edge, dist, labels, distType, maskSize, voronoiType_);

        if( voronoiType_ < 0 )
        {
            // begin "painting" the distance transform result
            dist *= 5000;
            pow(dist, 0.5, dist);

            Mat dist32s, dist8u1, dist8u2;

            dist.convertTo(dist32s, CV_32S, 1, 0.5);
            dist32s &= Scalar::all(255);

            dist32s.convertTo(dist8u1, CV_8U, 1, 0);
            dist32s *= -1;

            dist32s += Scalar::all(255);
            dist32s.convertTo(dist8u2, CV_8U);

            Mat planes[] = {dist8u1, dist8u2, dist8u2};
            merge(planes, 3, dist8u);
        }
        else
        {
            dist8u.create(labels.size(), CV_8UC3);
            for( int i = 0; i < labels.rows; i++ )
            {
                const int* ll = (const int*)labels.ptr(i);
                const float* dd = (const float*)dist.ptr(i);
                uchar* d = (uchar*)dist8u.ptr(i);
                for( int j = 0; j < labels.cols; j++ )
                {
                    int idx = ll[j] == 0 || dd[j] == 0 ? 0 : (ll[j]-1)%8 + 1;
                    float scale = 1.f/(1 + dd[j]*dd[j]*0.0004f);
                    int b = cvRound(colors[idx][0]*scale);
                    int g = cvRound(colors[idx][1]*scale);
                    int r = cvRound(colors[idx][2]*scale);
                    d[j*3] = (uchar)b;
                    d[j*3+1] = (uchar)g;
                    d[j*3+2] = (uchar)r;
                }
            }
        }
        res = dist8u;
        imshow(name_, dist8u);
        return res;
    }
    int maskSize0_;
    int voronoiType_;
    int edgeThresh_;
    int distType0_;
    int method_;
};

// apply:
// contours

/**
imgproc
CV_EXPORTS_W void medianBlur( InputArray src, OutputArray dst, int ksize );
CV_EXPORTS_W void GaussianBlur( InputArray src, OutputArray dst, Size ksize,
CV_EXPORTS_W void bilateralFilter( InputArray src, OutputArray dst, int d,
CV_EXPORTS_W void boxFilter( InputArray src, OutputArray dst, int ddepth,
CV_EXPORTS_W void sqrBoxFilter( InputArray src, OutputArray dst, int ddepth,
CV_EXPORTS_W void blur( InputArray src, OutputArray dst,
CV_EXPORTS_W void filter2D( InputArray src, OutputArray dst, int ddepth,
CV_EXPORTS_W void sepFilter2D( InputArray src, OutputArray dst, int ddepth,
CV_EXPORTS_W void Sobel( InputArray src, OutputArray dst, int ddepth,
CV_EXPORTS_W void Scharr( InputArray src, OutputArray dst, int ddepth,
CV_EXPORTS_W void Laplacian( InputArray src, OutputArray dst, int ddepth,
CV_EXPORTS_W void cornerMinEigenVal( InputArray src, OutputArray dst,
CV_EXPORTS_W void cornerHarris( InputArray src, OutputArray dst, int blockSize,
CV_EXPORTS_W void cornerEigenValsAndVecs( InputArray src, OutputArray dst,
CV_EXPORTS_W void preCornerDetect( InputArray src, OutputArray dst, int ksize,
CV_EXPORTS_W void erode( InputArray src, OutputArray dst, InputArray kernel,
CV_EXPORTS_W void dilate( InputArray src, OutputArray dst, InputArray kernel,
CV_EXPORTS_W void morphologyEx( InputArray src, OutputArray dst,
CV_EXPORTS_W void resize( InputArray src, OutputArray dst,
CV_EXPORTS_W void warpAffine( InputArray src, OutputArray dst,
CV_EXPORTS_W void warpPerspective( InputArray src, OutputArray dst,
CV_EXPORTS_W void remap( InputArray src, OutputArray dst,
                               OutputArray dstmap1, OutputArray dstmap2,
CV_EXPORTS_W void logPolar( InputArray src, OutputArray dst,
CV_EXPORTS_W void linearPolar( InputArray src, OutputArray dst,
CV_EXPORTS_W void warpPolar(InputArray src, OutputArray dst, Size dsize,
CV_EXPORTS_W void accumulate( InputArray src, InputOutputArray dst,
CV_EXPORTS_W void accumulateSquare( InputArray src, InputOutputArray dst,
                                     InputOutputArray dst, InputArray mask=noArray() );
CV_EXPORTS_W void accumulateWeighted( InputArray src, InputOutputArray dst,
CV_EXPORTS_W void createHanningWindow(OutputArray dst, Size winSize, int type);
CV_EXPORTS_W double threshold( InputArray src, OutputArray dst,
CV_EXPORTS_W void adaptiveThreshold( InputArray src, OutputArray dst,
CV_EXPORTS_W void pyrDown( InputArray src, OutputArray dst,
CV_EXPORTS_W void pyrUp( InputArray src, OutputArray dst,
CV_EXPORTS_W void undistort( InputArray src, OutputArray dst,
CV_EXPORTS_W void undistortPoints( InputArray src, OutputArray dst,
CV_EXPORTS_AS(undistortPointsIter) void undistortPoints( InputArray src, OutputArray dst,
                                   InputArray hist, OutputArray dst,
CV_EXPORTS_W void equalizeHist( InputArray src, OutputArray dst );
CV_EXPORTS_W void pyrMeanShiftFiltering( InputArray src, OutputArray dst,
CV_EXPORTS_AS(distanceTransformWithLabels) void distanceTransform( InputArray src, OutputArray dst,
CV_EXPORTS_W void distanceTransform( InputArray src, OutputArray dst,
CV_EXPORTS void blendLinear(InputArray src1, InputArray src2, InputArray weights1, InputArray weights2, OutputArray dst);
CV_EXPORTS_W void cvtColor( InputArray src, OutputArray dst, int code, int dstCn = 0 );
CV_EXPORTS_W void cvtColorTwoPlane( InputArray src1, InputArray src2, OutputArray dst, int code );
CV_EXPORTS_W void demosaicing(InputArray src, OutputArray dst, int code, int dstCn = 0);
CV_EXPORTS_W void applyColorMap(InputArray src, OutputArray dst, int colormap);
CV_EXPORTS_W void applyColorMap(InputArray src, OutputArray dst, InputArray userColor);

**/

/**
core

**/
class channel_filter : public itf_filter
{
public:
    channel_filter(const string& name) : itf_filter(name)
    {
        op_ = 0;
        channel_ = 0;
        threshold_ = 128;
        createTrackbar("threshold", name_, &threshold_, 255, itf_filter::update_, this);
        createTrackbar("(full,==,!=,<,>,<=,>=,&,^,|,&~)", name_, &op_, 10, itf_filter::update_, this);
        createTrackbar("channel (0-3)", name_, &channel_, 3, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res(image.size(), CV_8U);
        int ch[] = {std::min((image.type() >> CV_CN_SHIFT), channel_), 0};
        mixChannels(&image, 1, &res, 1, ch, 1);
        switch (op_)
        {
        case 1: res = res == threshold_; break;
        case 2: res = res != threshold_; break;
        case 3: res = res < threshold_; break;
        case 4: res = res > threshold_; break;
        case 5: res = res <= threshold_; break;
        case 6: res = res >= threshold_; break;
        case 7: res = res & threshold_; break;
        case 8: res = res ^ threshold_; break;
        case 9: res = res | threshold_; break;
        case 10: res = res & (~threshold_ & 0xff); break;
        }
        imshow(name_, res);
        return res;
    }
    int op_;
    int channel_;
    int threshold_;
};

class bgr2gray_filter : public itf_filter
{
public:
    bgr2gray_filter(const string& name) : itf_filter(name)
    {
        op_ = 0;
        threshold_ = 128;
        createTrackbar("threshold", name_, &threshold_, 255, itf_filter::update_, this);
        createTrackbar("(full,==,!=,<,>,<=,>=,&,^,|,&~)", name_, &op_, 10, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        cvtColor(image, res, CV_BGR2GRAY);
        switch (op_)
        {
        case 1: res = res == threshold_; break;
        case 2: res = res != threshold_; break;
        case 3: res = res < threshold_; break;
        case 4: res = res > threshold_; break;
        case 5: res = res <= threshold_; break;
        case 6: res = res >= threshold_; break;
        case 7: res = res & threshold_; break;
        case 8: res = res ^ threshold_; break;
        case 9: res = res | threshold_; break;
        case 10: res = res & (~threshold_ & 0xff); break;
        }
        imshow(name_, res);
        return res;
    }
    int op_;
    int threshold_;
};

class range_filter : public itf_filter
{
public:
    range_filter(const string& name) : itf_filter(name)
    {
        h1_ = s1_ = v1_ = 0;
        h2_ = s2_ = v2_ = 255;
        method_ = inv_ = 0;
        createTrackbar("h1(b,h)", name_, &h1_, 255, itf_filter::update_, this);
        createTrackbar("h2(b,h)", name_, &h2_, 255, itf_filter::update_, this);
        createTrackbar("s1(g,l)", name_, &s1_, 255, itf_filter::update_, this);
        createTrackbar("s2(g,l)", name_, &s2_, 255, itf_filter::update_, this);
        createTrackbar("v1(r,s)", name_, &v1_, 255, itf_filter::update_, this);
        createTrackbar("v2(r,s)", name_, &v2_, 255, itf_filter::update_, this);
        createTrackbar("hsv,bgr,hls", name_, &method_, 2, itf_filter::update_, this);
        createTrackbar("invert mask", name_, &inv_, 1, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        Mat hsv, mask;
        if (0 == method_)
            cvtColor(image, hsv, CV_BGR2HSV);
        else if (2 == method_)
            cvtColor(image, hsv, CV_BGR2HLS);
        else
            hsv = image;
        inRange(hsv,
                Scalar(min(h1_, h2_), min(s1_, s2_), min(v1_, v2_)),
                Scalar(max(h1_, h2_), max(s1_, s2_), max(v1_, v2_)),
                mask);
        bitwise_and(image, image, res, (inv_)?~mask:mask);
        imshow(name_, res);
        return res;
    }
    int h1_, h2_, s1_, s2_, v1_, v2_;
    int method_;
    int inv_;
};

class colormap_filter : public itf_filter
{
public:
    colormap_filter(const string& name) : itf_filter(name)
    {
        type_ = 0;
        createTrackbar("type:\n"
                       "AUTUMN=0, BONE, JET, WINTER, RAINBOW,\n"
                       "OCEAN=5, SUMMER, SPRING, COOL, HSV,\n"
                       "PINK=10, HOT, PARULA, MAGMA, INFERNO,\n"
                       "PLASMA=15, VIRIDIS, CIVIDIS, TWILIGHT,\n"
                       "TWILIGHT_SHIFTED=19, TURBO, DEEPGREEN", name_,
                       &type_, 21, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        applyColorMap(image, res, type_);
        imshow(name_, res);
        return res;
    }
    int type_;
};

// opencv/samples/cpp/digits.cpp
class deskew_filter : public itf_filter
{
public:
    deskew_filter(const string& name) : itf_filter(name)
    {
        sz_ = 20;
        createTrackbar("size", name_, &sz_, 10000, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        if (!image.empty())
            setTrackbarMax("size", name_, max(image.cols, image.rows));
        sz_ = max(1, sz_);
        Moments m = moments(image);
        if (abs(m.mu02) < 0.01)
        {
            res = image.clone();
        }
        else
        {
            float skew = (float)(m.mu11 / m.mu02);
            float M_vals[2][3] = {{1, skew, -0.5f * sz_ * skew}, {0, 1, 0}};
            Mat M(Size(3, 2), CV_32F);

            for (int i = 0; i < M.rows; i++)
            {
                for (int j = 0; j < M.cols; j++)
                {
                    M.at<float>(i, j) = M_vals[i][j];
                }
            }

            warpAffine(image, res, M, Size(sz_, sz_), WARP_INVERSE_MAP | INTER_LINEAR);
        }
        imshow(name_, res);
        return res;
    }
    int sz_;
};

// op
class cut_filter : public itf_filter
{
public:
    cut_filter(const string& name) : itf_filter(name)
    {
        use_save_ = true;
        state_ = cut_IDLE;
        setMouseCallback(name_, on_mouse, this);
    }
    cut_filter& apply(function<void(Mat,Mat)> f)
    {
        apply_ = f;
        return *this;
    }
protected:
    static void on_mouse(int event, int x, int y, int flags, void* ctx)
    {
        ((cut_filter*)ctx)->_on_mouse(event, x, y, flags, ctx);
    }
    void _on_mouse(int event, int x, int y, int flags, void* ctx)
    {
        ostringstream os;
        switch (event)
        {
        case EVENT_RBUTTONDBLCLK:
            if (cut_FIN == state_)
            {
                Mat dst;
                cutrect_ = rect_;
                curve_(cutrect_).copyTo(dst);
                os << "cut_"
                         << rect_.x << "_"
                         << rect_.y << "_"
                         << rect_.width << "_"
                         << rect_.height;
                if (EVENT_FLAG_CTRLKEY & flags)
                    os << ".bmp";
                else if (EVENT_FLAG_SHIFTKEY & flags)
                    os << ".jpg";
                else
                    os << ".png";
                if (use_save_)
                    imwrite(os.str(), dst);
                if (apply_)
                {
                    cut_ = dst;
                    apply_(curve_, cut_);
                }
            }
            state_ = cut_SAVE;
            break;
        case EVENT_LBUTTONDOWN:
            state_ = cut_EDIT;
            rect_.x = max(0, x);
            rect_.y = max(0, y);
            break;
        case EVENT_MOUSEMOVE:
            if (cut_EDIT == state_)
            {
                rect_.width = max(0, x) - rect_.x;
                rect_.height = max(0, y) - rect_.y;
            }
            break;
        case EVENT_LBUTTONUP:
            if (cut_EDIT == state_)
            {
                if (0 == rect_.width)
                    rect_.width = 4;
                if (0 == rect_.height)
                    rect_.height = 4;
                if (rect_.width < 0)
                {
                    rect_.x += rect_.width;
                    rect_.width -= rect_.width + rect_.width;
                }
                if (rect_.height < 0)
                {
                    rect_.y += rect_.height;
                    rect_.height -= rect_.height + rect_.height;
                }
                if (rect_.y + rect_.height > curve_.rows)
                {
                    rect_.height = curve_.rows - rect_.y;
                }
                if (rect_.x + rect_.width > curve_.cols)
                {
                    rect_.width = curve_.cols - rect_.x;
                }
                state_ = cut_FIN;
            }
            break;
        }

        if (state_ > cut_IDLE)
        {
            Mat show;
            curve_.copyTo(show);
            if (state_ < cut_RESET)
                rectangle(show, Point( rect_.x, rect_.y ), Point(rect_.x + rect_.width, rect_.y + rect_.height ), Scalar(0, 255, 0), 1);
            else
            {
                if (cut_SAVE == state_)
                {
                    rectangle(show, Point( cutrect_.x, cutrect_.y ), Point(cutrect_.x + cutrect_.width, cutrect_.y + cutrect_.height ), Scalar(0, 0, 255), 1);
                    putText(show, (os << " saved!", os).str(), Point(0, 20), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 185, 0));
                }
                state_ = cut_IDLE;
            }
            imshow(name_, show);
        }
    }
    virtual Mat _filter(Mat& image)
    {
        curve_ = image;
        state_ = cut_IDLE;
        //rect_ = Rect();
        if (cutrect_.empty())
            imshow(name_, curve_);
        else
        {
            Mat show;
            curve_.copyTo(show);
            rectangle(show, Point(cutrect_.x, cutrect_.y ), Point(cutrect_.x + cutrect_.width, cutrect_.y + cutrect_.height ), Scalar(0, 0, 255), 1);
            imshow(name_, show);
        }
        if (apply_)
            apply_(image, cut_);
        return image;
    }
    Rect fullrect_;
    Rect rect_;
    Rect cutrect_;
    enum {
        cut_IDLE,
        cut_EDIT,
        cut_FIN,
        cut_RESET,
        cut_SAVE,
    } state_;
    Mat curve_;
    Mat cut_;
    bool use_save_;
    function<void(Mat,Mat)> apply_;
};

// cut2
// generate pos/$time.jpg and neg/$time.jpg
// append info to pos/pos.txt and neg/neg.txt
class cut2_filter : public itf_filter
{
public:
    cut2_filter(const string& name) : itf_filter(name)
    {
        use_save_ = true;
        state_ = cut_IDLE;
        setMouseCallback(name_, on_mouse, this);
    }
protected:
    static void on_mouse(int event, int x, int y, int flags, void* ctx)
    {
        ((cut2_filter*)ctx)->_on_mouse(event, x, y, flags, ctx);
    }
    void save_pos(const string& name)
    {
        imwrite("pos/" + name, curve_);
        ofstream fout;
        fout.open("pos/pos.txt", ios_base::out|ios_base::app);
        fout << "pos/" << name << " ";
        fout << posrect_.size() << " ";
        for_each(posrect_.begin(), posrect_.end(),
                 [&](Rect& rect){
                     fout << rect.x << " "
                        << rect.y << " "
                        << rect.width << " "
                        << rect.height << " ";
                 });
        fout << "\n";
        fout.close();
    }
    void save_neg(const string& name)
    {
        Mat neg = curve_.clone();
        Mat element = getStructuringElement(MORPH_RECT, Size(21, 21), Point(10, 10));
        Mat morph;
        morphologyEx(curve_, morph, MORPH_OPEN, element);
        for_each(posrect_.begin(), posrect_.end(),
                 [&](Rect& rect){
                     //morph(rect).copyTo(neg(rect));
                     ((Mat)Mat::zeros(rect.size(), neg.type())).copyTo(neg(rect));
                 });
        imwrite("neg/" + name, neg);
        ofstream fout;
        fout.open("neg/neg.txt", ios_base::out|ios_base::app);
        fout << "neg/" << name << "\n";
        fout.close();
    }
    void _on_mouse(int event, int x, int y, int flags, void* ctx)
    {
        ostringstream os;
        switch (event)
        {
        case EVENT_MBUTTONDOWN:
            state_ = cut_IDLE;
            rect_ = Rect();
            posrect_.clear();
            break;
        case EVENT_RBUTTONDBLCLK:
            if (!posrect_.empty())
            {
                Mat dst;
                cutrect_ = rect_;
                curve_(cutrect_).copyTo(dst);
                time_t ts = time(0);
                os << "cascade_"
                         << ts;
                if (EVENT_FLAG_CTRLKEY & flags)
                    os << ".bmp";
                else if (EVENT_FLAG_SHIFTKEY & flags)
                    os << ".jpg";
                else
                    os << ".png";
                save_pos(os.str());
                save_neg(os.str());

                posrect_.clear();
                //rect_ = Rect();
                saved_name_ = os.str();
            }
            state_ = cut_SAVE;
            break;
        case EVENT_LBUTTONDOWN:
            state_ = cut_PREEDIT;
            rect_.x = max(0, x);
            rect_.y = max(0, y);
            if (EVENT_FLAG_CTRLKEY & flags)
            {
                state_ = cut_EDIT;
                rect_.width = 0;
                rect_.height = 0;
            }
            break;
        case EVENT_MOUSEMOVE:
            if (cut_EDIT == state_
                && (EVENT_FLAG_CTRLKEY & flags))
            {
                rect_.width = max(0, x) - rect_.x;
                rect_.height = max(0, y) - rect_.y;
            }
            break;
        case EVENT_LBUTTONDBLCLK:
            if (!(EVENT_FLAG_CTRLKEY & flags)
                && rect_.width && rect_.height)
            {
                rect_.x = max(0, x);
                rect_.y = max(0, y);
                if (rect_.y + rect_.height > curve_.rows)
                {
                    rect_.height = curve_.rows - rect_.y;
                }
                if (rect_.x + rect_.width > curve_.cols)
                {
                    rect_.width = curve_.cols - rect_.x;
                }
                posrect_.push_back(rect_);
                state_ = cut_FIN;
            }
            break;
        case EVENT_LBUTTONUP:
            if (cut_EDIT == state_
                && (EVENT_FLAG_CTRLKEY & flags))
            {
                if (0 == rect_.width)
                    rect_.width = 4;
                if (0 == rect_.height)
                    rect_.height = 4;
                if (rect_.width < 0)
                {
                    rect_.x += rect_.width;
                    rect_.width -= rect_.width + rect_.width;
                }
                if (rect_.height < 0)
                {
                    rect_.y += rect_.height;
                    rect_.height -= rect_.height + rect_.height;
                }
                if (rect_.y + rect_.height > curve_.rows)
                {
                    rect_.height = curve_.rows - rect_.y;
                }
                if (rect_.x + rect_.width > curve_.cols)
                {
                    rect_.width = curve_.cols - rect_.x;
                }
                posrect_.push_back(rect_);
                state_ = cut_FIN;
            }
            break;
        }

        /**
        if (state_ > cut_IDLE)
        {
            Mat show;
            curve_.copyTo(show);
            if (state_ < cut_RESET)
                rectangle(show, Point( rect_.x, rect_.y ), Point(rect_.x + rect_.width, rect_.y + rect_.height ), Scalar(0, 255, 0), 1);
            else
            {
                if (cut_SAVE == state_)
                {
                    putText(show, (os << " saved!", os).str(), Point(0, 20), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 185, 0));
                }
                state_ = cut_IDLE;
            }
            for_each(posrect_.begin(), posrect_.end(),
                     [&](Rect& rect){
                        rectangle(show, Point( rect.x, rect.y ), Point(rect.x + rect.width, rect.y + rect.height ), Scalar(0, 0, 255), 1);
                     });
            imshow(name_, show);
        }
        */
        show();
        if (cut_SAVE == state_)
            state_ = cut_IDLE;
    }
    void show()
    {
        if (posrect_.empty() && state_ == cut_IDLE)
        {
            imshow(name_, curve_);
            return;
        }
        Mat show;
        curve_.copyTo(show);
        if (state_ > cut_IDLE && state_ < cut_RESET)
        {
            rectangle(show, Point( rect_.x, rect_.y ), Point(rect_.x + rect_.width, rect_.y + rect_.height ), Scalar(0, 255, 0), 1);
        }
        if (cut_SAVE == state_)
        {
            putText(show, saved_name_, Point(0, 20), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 185, 0));
        }
        for_each(posrect_.begin(), posrect_.end(),
                     [&](Rect& rect){
                        rectangle(show, Point( rect.x, rect.y ), Point(rect.x + rect.width, rect.y + rect.height ), Scalar(0, 0, 255), 1);
                     });
        imshow(name_, show);
    }
    virtual Mat _filter(Mat& image)
    {
        curve_ = image;
        state_ = cut_IDLE;
        //rect_ = Rect();
        /**
        if (cutrect_.empty())
            imshow(name_, curve_);
        else
        {
            Mat show;
            curve_.copyTo(show);
            rectangle(show, Point(cutrect_.x, cutrect_.y ), Point(cutrect_.x + cutrect_.width, cutrect_.y + cutrect_.height ), Scalar(0, 0, 255), 1);
            imshow(name_, show);
        }
        */
        show();
        return image;
    }
    Rect fullrect_;
    Rect rect_;
    Rect cutrect_;
    vector<Rect> posrect_;
    enum {
        cut_IDLE,
        cut_PREEDIT,
        cut_EDIT,
        cut_FIN,
        cut_RESET,
        cut_SAVE,
    } state_;
    Mat curve_;
    string saved_name_;
    bool use_save_;
};

class crop_filter : public cut_filter
{
public:
    crop_filter(const string& name) : cut_filter(name)
    {
        use_save_  = false;
        apply([this](Mat, Mat crop){
              if (state_ == cut_FIN)
              {
                  update_next_(crop);
                  graph_->update_origin(crop);
              }
        });
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res = image;
        cut_filter::_filter(image);

        if (!cutrect_.empty() && !cut_.empty())
        {
            curve_(cutrect_).copyTo(cut_);
            res = cut_;
            graph_->update_origin(cut_);
        }
        return res;
    }
    Mat crop_;
};

class zoom_filter : public itf_filter
{
public:
    zoom_filter(const string& name) : itf_filter(name)
    {
        x_ = 100;
        y_ = 100;
        inter_ = 1;
        createTrackbar("zoom x(%)", name_, &x_, 400, itf_filter::update_, this);
        createTrackbar("zoom y(%)", name_, &y_, 400, itf_filter::update_, this);
        createTrackbar("interpola:\n"
            "INTER_NEAREST        = 0,\n"
            "INTER_LINEAR         = 1,\n"
            "INTER_CUBIC          = 2,\n"
            "INTER_AREA           = 3,\n"
            "INTER_LANCZOS4       = 4,\n"
            "INTER_LINEAR_EXACT = 5, \n"
            "INTER_NEAREST_EXACT  = 6,\n"
            "INTER_MAX            = 7,\n", name_,
            &inter_, 7, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        setTrackbarMin("zoom x(%)", name_, 50);
        setTrackbarMin("zoom y(%)", name_, 50);
        if (x_ != 100 || y_ != 100)
            resize(image, res, Size(), x_/100., y_/100., inter_);
        else
            res = image;
        graph_->update_origin(res);
        imshow(name_, res);
        return res;
    }
    int x_;
    int y_;
    int inter_;
};

// apply
class contours_filter : public itf_filter
{
public:
    contours_filter(const string& name) : itf_filter(name)
    {
        threshold_ = 20;
        retr_ = 1;
        showpoly_ = 1;
        showcontours_ = 1;
        createTrackbar("threshold*.001", name_, &threshold_, 200, itf_filter::update_, this);
        createTrackbar("show poly(OFF/ON)", name_, &showpoly_, 1, itf_filter::update_, this);
        createTrackbar("show contours(OFF/ON)", name_, &showcontours_, 1, itf_filter::update_, this);
        createTrackbar("RETR:\n"
            "0 - RETR_EXTERNAL \n"
            "1 - RETR_LIST\n"
            "2 - RETR_CCOMP\n"
            "3 - RETR_TREE\n"
            "4 - RETR_FLOODFILL\n", name_,
            &retr_, 4, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        vector<vector<Point> > contours;
        findContours(image, contours, retr_, CHAIN_APPROX_SIMPLE);
        vector<Point> approx;

        Mat show = graph_->origin().clone();
        // test each contour
        if (showpoly_)
            for( size_t i = 0; i < contours.size(); i++ )
            {
                approxPolyDP(contours[i], approx, arcLength(contours[i], true)*threshold_/1000., true);
                polylines(show, approx, true, Scalar(255, 0, 0), 2, LINE_AA);
            }
        if (showcontours_)
            drawContours(show, contours, -1, Scalar(0, 255, 0), 1, LINE_AA);
        imshow(name_, show);
        return image;
    }
    int threshold_;
    int retr_;
    int showpoly_;
    int showcontours_;
};

class cascade_filter : public itf_filter
{
public:
    cascade_filter(const string& name) : itf_filter(name)
    {
        algo1_ = algo2_ = algo3_ = algo4_ = 0;
        algo2_ = 1;
        x_ = 20;
        y_ = 20;
        scalefactor_ = 15;
        minneighbros_ = 3;
        cascade_.load(samples::findFile("cascade/cascade.xml"));
        createTrackbar("scalefactor*.01 + 1", name_, &scalefactor_, 100, itf_filter::update_, this);
        createTrackbar("min neighbros", name_, &minneighbros_, 10, itf_filter::update_, this);
        createTrackbar("DO_CANNY_PRUNING   (OFF/ON)", name_, &algo1_, 1, itf_filter::update_, this);
        createTrackbar("SCALE_IMAGE        (OFF/ON)", name_, &algo2_, 1, itf_filter::update_, this);
        createTrackbar("FIND_BIGGEST_OBJECT(OFF/ON)", name_, &algo3_, 1, itf_filter::update_, this);
        createTrackbar("DO_ROUGH_SEARCH    (OFF/ON)", name_, &algo4_, 1, itf_filter::update_, this);
        createTrackbar("x", name_, &x_, 100, itf_filter::update_, this);
        createTrackbar("y", name_, &y_, 100, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res = image;
        if (!cascade_.empty())
        {
            vector<Rect> objs;
            Mat show = res.clone();
            cascade_.detectMultiScale(image, objs, scalefactor_/100.+1, minneighbros_,
                                      (algo1_ << 0)|(algo2_ << 1)|(algo2_ << 2)|(algo3_ << 3),
                                      Size(x_, y_));
            int i = 0;
            for_each(objs.begin(), objs.end(),
                     [&](Rect& rect){
                        static const Scalar colors[] =
                        {
                            Scalar(0,0,0),
                            Scalar(255,0,0),
                            Scalar(255,128,0),
                            Scalar(255,255,0),
                            Scalar(0,255,0),
                            Scalar(0,128,255),
                            Scalar(0,255,255),
                            Scalar(0,0,255),
                            Scalar(255,0,255)
                        };
                        rectangle(show, rect, colors[++i%8], 1, LINE_AA);
                     });
            imshow(name_, show);
        }

        return res;
    }
    int x_;
    int y_;
    int scalefactor_;
    int minneighbros_;
    int algo1_, algo2_, algo3_, algo4_;
    CascadeClassifier cascade_;
};

class exec_filter : public itf_filter
{
public:
    exec_filter(const string& name) : itf_filter(name)
    {
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        if (apply_)
            apply_(image);
        imshow(name_, image);
        return image;
    }
    function<void(Mat)> apply_;
};

itf_filter* createFilter(const char* filter, const string& name)
{
#define BRANCH(nameX)   \
    if (0 == strcasecmp(#nameX, filter))   \
    {   \
        return (itf_filter*)new nameX##_filter(name);  \
    }
    if (0 == strcasecmp("threshold", filter))
    {
        return (itf_filter*)new threshold_filter(name);
    }
    if (0 == strcasecmp("morphology", filter))
    {
        return (itf_filter*)new morphology_filter(name);
    }
    BRANCH(Canny);
    BRANCH(HoughLines);
    BRANCH(medianBlur);
    BRANCH(GaussianBlur);
    BRANCH(blur);
    BRANCH(bilateral);
    BRANCH(box);
    BRANCH(sqrBox);
    BRANCH(Sobel);
    BRANCH(Scharr);
    BRANCH(Laplacian);
    BRANCH(cornerMinEigenVal);
    BRANCH(cornerHarris);
    BRANCH(cornerEigenValsAndVecs);
    BRANCH(preCornerDetect);
    BRANCH(warpAffine);
    BRANCH(warpPerspective);
    BRANCH(warpPolar);
    BRANCH(pyrDown);
    BRANCH(pyrUp);

    BRANCH(channel);
    BRANCH(bgr2gray);
    BRANCH(range);
    BRANCH(colormap);
    BRANCH(cut);
    BRANCH(cut2);
    BRANCH(crop);
    BRANCH(zoom);

    BRANCH(deskew);
    BRANCH(dem);
    BRANCH(distrans);

    BRANCH(contours);
    BRANCH(cascade);
    return NULL;
}

void filter_graph::open(const string& cmd)
{
    string cmds = cmd;
    {
        char* p = (char*)&cmds.at(0);
        while (*p)
        {
            if (*p == ',')
                *p = '\0';
            ++p;
        }
    }
    const char* p = cmds.c_str();
    const char* ep = p + cmds.size();
    while (p < ep)
    {
        sptr_filter filter(createFilter(p, p));
        if (filter)
            push(filter);
        else
            cout << "unknown " << p << endl;
        p += strlen(p) + 1;
    }
}

} // end ns cvtool

} // end ns zhelper

#endif // ZCVTOOL_HELPER__H_

