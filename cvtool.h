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
#include <opencv2/features2d.hpp>
#include <opencv2/cvconfig.h>

//#include <opencv2/imgcodecs.hpp>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <algorithm>
using namespace cv;
using namespace std;

#if (CV_VERSION_MAJOR >= 4)
#define CV_BGR2GRAY cv::COLOR_BGR2GRAY
#define CV_BGR2HSV cv::COLOR_BGR2HSV
#define CV_BGR2HLS cv::COLOR_BGR2HLS
#define CV_TM_SQDIFF TM_SQDIFF
#define CV_TM_CCORR_NORMED TM_CCORR_NORMED
#endif

#if defined(OPENCV_ENABLE_NONFREE) || CV_VERSION_MAJOR >= 4
#define NON_FREE
#endif

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
	itf_filter(const string& name, const string& comment) : num_(++snum_)
	{
		ostringstream os;
		os << num_ << ": " << name << comment << " -=>@github.com/bbqz007";
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
	void bringTop()
	{
		setWindowProperty(name_, 5, 0);
	}
protected:
    static const Scalar& next_color(bool reset = false)
    {
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
        static const int limits = sizeof(colors) / sizeof(Scalar);
        static int i = 0;
        if (reset)
            i = 0;
        return colors[++i%limits];
    }
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
	void bringTop(int i)
	{
		if (i < filters_.size())
			filters_[i]->bringTop();
	}
	void bringTop()
	{
		if (!filters_.empty())
			filters_.back()->bringTop();
	}
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
        rho_ = 10;
        theta_ = 1;
        threshval_ = 150;
        createTrackbar("rho*.1", name_, &rho_, 100, itf_filter::update_, this);
        createTrackbar("theta angle", name_, &theta_, 180, itf_filter::update_, this);
        createTrackbar("threshval", name_, &threshval_, 1000, itf_filter::update_, this);
        setTrackbarMin("theta angle", name_, 1);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res = image;
        vector<Vec2f> lines;
        HoughLines(image, lines, rho_*.1, theta_*(CV_PI/180), threshval_);
        Mat show = graph_->origin().clone();
        for( size_t i = 0; i < lines.size(); i++ )
        {
            float rho = lines[i][0], theta = lines[i][1];
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));
            line(show, pt1, pt2, Scalar(0,0,255), 2, LINE_AA);
        }
        imshow(name_, show);
        return res;
    }
    int rho_;
    int theta_;
    int threshval_;
};

class HoughLinesP_filter : public itf_filter
{
public:
    HoughLinesP_filter(const string& name) : itf_filter(name)
    {
        rho_ = 10;
        theta_ = 1;
        threshval_ = 50;
        createTrackbar("rho*.1", name_, &rho_, 100, itf_filter::update_, this);
        createTrackbar("theta angle", name_, &theta_, 180, itf_filter::update_, this);
        createTrackbar("threshval", name_, &threshval_, 1000, itf_filter::update_, this);
        setTrackbarMin("theta angle", name_, 1);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res = image;
        vector<Vec4i> lines;
        HoughLinesP(image, lines, rho_*.1, theta_*(CV_PI/180), threshval_, 50, 10);
        Mat show = graph_->origin().clone();
        for( size_t i = 0; i < lines.size(); i++ )
        {
            Vec4i l = lines[i];
            line(show, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 2, LINE_AA);
        }
        imshow(name_, show);
        return res;
    }
    int rho_;
    int theta_;
    int threshval_;
};

class HoughCircles_filter : public itf_filter
{
public:
    HoughCircles_filter(const string& name) : itf_filter(name)
    {
        dist_ = 16;
        radius1_ = 1;
        radius2_ = 30;
        createTrackbar("distance", name_, &dist_, 32, itf_filter::update_, this);
        createTrackbar("radius (bound 1)", name_, &radius1_, 100, itf_filter::update_, this);
        createTrackbar("radius (bound 2)", name_, &radius2_, 100, itf_filter::update_, this);
        setTrackbarMin("radius (bound 1)", name_, 1);
        setTrackbarMin("radius (bound 2)", name_, 1);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res = image;
        vector<Vec3f> circles;
        HoughCircles(image, circles, HOUGH_GRADIENT, 1, image.rows/max(dist_, 1), 100, 30,
                     min(radius1_, radius2_), max(radius1_, radius2_));
        Mat show = graph_->origin().clone();
        for( size_t i = 0; i < circles.size(); i++ )
        {
            Vec3i c = circles[i];
            Point center = Point(c[0], c[1]);
            // circle center
            circle(show, center, 1, Scalar(0,100,100), 1, LINE_AA);
            // circle outline
            int radius = c[2];
            circle(show, center, radius, Scalar(255,0,255), 1, LINE_AA);
        }
        imshow(name_, show);
        return res;
    }
    int dist_;
    int radius1_;
    int radius2_;
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
        threshold1_ = 0;
        threshold2_ = 0;
        createTrackbar("ksize|1", name_, &ksize_, 11, itf_filter::update_, this);
        //createTrackbar("threshold*.001(bound 1)", name_, &threshold1_, 1000, itf_filter::update_, this);
        //createTrackbar("threshold*.001(bound 2)", name_, &threshold2_, 1000, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        preCornerDetect(image, res, ksize_|1);
        double minV, maxV;
        minMaxLoc(res, &minV, &maxV);
        setTrackbarMin("threshold*.001(bound 1)", name_, int(minV*1000));
        setTrackbarMin("threshold*.001(bound 2)", name_, int(minV*1000));
        setTrackbarMax("threshold*.001(bound 1)", name_, int(maxV*1000));
        setTrackbarMax("threshold*.001(bound 2)", name_, int(maxV*1000));
        //Mat show = (res >= (min(threshold1_, threshold2_) / 1000.)) <= (max(threshold1_, threshold2_) / 1000.);
        imshow(name_, res);
        return res;
    }
    int ksize_;
    int threshold1_;
    int threshold2_;
};

class norm_minmax_filter : public itf_filter
{
public:
    norm_minmax_filter(const string& name) : itf_filter(name)
    {
        op_ = 0;
        switch_ = 1;
        threshold_ = 128;
        createTrackbar("threshold*.001", name_, &threshold_, 1000, itf_filter::update_, this);
        createTrackbar("(full,<,>,<=,>=)", name_, &op_, 4, itf_filter::update_, this);
        createTrackbar("normalize (OFF/ON)", name_, &op_, 1, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        if (switch_)
            normalize(image, res, 0, 1, NORM_MINMAX);
        switch (op_)
        {
        case 1: res = res < threshold_*.001; break;
        case 2: res = res > threshold_*.001; break;
        case 3: res = res <= threshold_*.001; break;
        case 4: res = res >= threshold_*.001; break;
        }
        imshow(name_, res);
        return res;
    }
    int threshold_;
    int op_;
    int switch_;
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
        threshold1_ = 128;
        threshold2_ = 128;
        createTrackbar("threshold 1", name_, &threshold1_, 255, itf_filter::update_, this);
        createTrackbar("threshold 2", name_, &threshold2_, 255, itf_filter::update_, this);
        createTrackbar("(full,range,<|>,&,^,|,&~)", name_, &op_, 6, itf_filter::update_, this);
        createTrackbar("channel (0-3)", name_, &channel_, 3, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res(image.size(), CV_8U);
        Mat mask;
        int ch[] = {std::min((image.type() >> CV_CN_SHIFT), channel_), 0};
        mixChannels(&image, 1, &res, 1, ch, 1);
        switch (op_)
        {
        case 1:
            inRange(res, Scalar(min(threshold1_, threshold2_)), Scalar(max(threshold1_, threshold2_)), mask);
            bitwise_and(mask, res, res);
            break;
        case 2:
            inRange(res, Scalar(min(threshold1_, threshold2_)), Scalar(max(threshold1_, threshold2_)), mask);
            bitwise_and(~mask, res, res);
            break;
        case 3: res = res & threshold1_; break;
        case 4: res = res ^ threshold1_; break;
        case 5: res = res | threshold1_; break;
        case 6: res = res & (~threshold1_ & 0xff); break;
        }
        imshow(name_, res);
        return res;
    }
    int op_;
    int channel_;
    int threshold1_, threshold2_;
};

class bgr2gray_filter : public itf_filter
{
public:
    bgr2gray_filter(const string& name) : itf_filter(name)
    {
        op_ = 0;
        threshold1_ = 128;
        threshold2_ = 128;
        createTrackbar("threshold 1", name_, &threshold1_, 255, itf_filter::update_, this);
        createTrackbar("threshold 2", name_, &threshold2_, 255, itf_filter::update_, this);
        createTrackbar("(full,range,<|>,&,^,|,&~)", name_, &op_, 6, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res;
        Mat mask;
        cvtColor(image, res, CV_BGR2GRAY);
        switch (op_)
        {
        case 1:
            inRange(res, Scalar(min(threshold1_, threshold2_)), Scalar(max(threshold1_, threshold2_)), mask);
            bitwise_and(mask, res, res);
            break;
        case 2:
            inRange(res, Scalar(min(threshold1_, threshold2_)), Scalar(max(threshold1_, threshold2_)), mask);
            bitwise_and(~mask, res, res);
            break;
        case 3: res = res & threshold1_; break;
        case 4: res = res ^ threshold1_; break;
        case 5: res = res | threshold1_; break;
        case 6: res = res & (~threshold1_ & 0xff); break;
        }
        imshow(name_, res);
        return res;
    }
    int op_;
    int threshold1_, threshold2_;
};

class gray2mask_filter : public itf_filter
{
public:
    gray2mask_filter(const string& name) : itf_filter(name)
    {
        op_ = 0;
        threshold_ = 128;
        createTrackbar("threshold", name_, &threshold_, 255, itf_filter::update_, this);
        createTrackbar("(full,==,!=,<,>,<=,>=)", name_, &op_, 6, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res = image;
        switch (op_)
        {
        case 1: res = res == threshold_; break;
        case 2: res = res != threshold_; break;
        case 3: res = res < threshold_; break;
        case 4: res = res > threshold_; break;
        case 5: res = res <= threshold_; break;
        case 6: res = res >= threshold_; break;
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
		method_ = inv_ = mask_only_ = 0;
        createTrackbar("h1(b,h)", name_, &h1_, 255, itf_filter::update_, this);
        createTrackbar("h2(b,h)", name_, &h2_, 255, itf_filter::update_, this);
        createTrackbar("s1(g,l)", name_, &s1_, 255, itf_filter::update_, this);
        createTrackbar("s2(g,l)", name_, &s2_, 255, itf_filter::update_, this);
        createTrackbar("v1(r,s)", name_, &v1_, 255, itf_filter::update_, this);
        createTrackbar("v2(r,s)", name_, &v2_, 255, itf_filter::update_, this);
        createTrackbar("hsv,bgr,hls", name_, &method_, 2, itf_filter::update_, this);
        createTrackbar("invert mask", name_, &inv_, 1, itf_filter::update_, this);
		createTrackbar("mask/no mask/mask only", name_, &mask_only_, 2, itf_filter::update_, this);
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
		if (mask_only_ == 2)
			res = (inv_) ? ~mask : mask;
		else if (mask_only_ == 0)
			bitwise_and(image, image, res, (inv_) ? ~mask : mask);
		else
			res = hsv;
        imshow(name_, res);
        return res;
    }
    int h1_, h2_, s1_, s2_, v1_, v2_;
    int method_;
    int inv_;
	int mask_only_;
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

/**
feature
    Feature2D is kind of Algorithm,
        has two main functions, detect and compute.
        1. detect corners KeyPoints
        2. compute description Mat(rix).
    A Feature2D backend may be detector or extractor or both.
    SIFT, ORB, BRISK, KAZE, AKAZE, MSER.
    FastFeatureDetector, AgastFeatureDetector, GFTTDetector, SimpleBlobDetector.
    AffineFeature implementing the wrapper which makes detectors and extractors to be affine invariant.
**/
class feature_filter : public itf_filter
{
public:
    feature_filter(const string& name) : itf_filter(name)
    {
        curfeat_ = -1;
        feature_ = 0;
        affine_ = 1;
        curaff_ = affine_;
        limits_ = 0;
        createTrackbar("feat count", name_, &limits_, INT_MAX, itf_filter::update_, this);
        createTrackbar("feature:\nsift:0\norb:1\nbrisk:2\n"
                       "kaze:3\nakaze:4\nmser:5\n"
                       "fast:6\nagast:7\ngftt:8\nblob:9\n",
                        name_, &feature_, 9, itf_filter::update_, this);
        createTrackbar("Affine (OFF/ON)", name_, &affine_, 1, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res = image;

        bool changed = feature_ != curfeat_ || affine_ != curaff_ ;
        curaff_ = affine_;
        if (feature_ != curfeat_)
        {
            
			switch (feature_)
            {
            // both detector and extractor
#ifdef NON_FREE
            case 0: backend_ = SIFT::create(); break;
            case 1: backend_ = ORB::create(); break;
            case 2: backend_ = BRISK::create(); break;
            case 3: backend_ = KAZE::create(); break;
            case 4: backend_ = AKAZE::create(); break;
            // just detector
            case 5: backend_ = MSER::create(); break;
            case 6: backend_ = FastFeatureDetector::create(); break;
            case 7: backend_ = AgastFeatureDetector::create(); break;
            case 8: backend_ = GFTTDetector::create(); break;
            case 9: backend_ = SimpleBlobDetector::create(); break;
#endif
			}
#ifdef NON_FREE
            ext_ = AffineFeature::create(backend_);
#endif
            curfeat_ = feature_;
        }


        vector<KeyPoint> kp1;
        Mat desc1;
        Mat show = image.clone();
        //ext_->detectAndCompute(image, Mat(), kp1, desc1);
		if (affine_ && feature_ < 5)
#ifdef NON_FREE
			ext_->detect(image, kp1);
#else
			;
#endif
        else
            backend_->detect(image, kp1);
        setTrackbarMax("feat count", name_, kp1.size());
        setTrackbarPos("feat count", name_, kp1.size());
        if (changed)
            limits_ = kp1.size();
        next_color(true);
        for_each(kp1.begin(), kp1.begin() + limits_,
                 [&](KeyPoint& kp) {
                    circle(show, kp.pt, 3, next_color());
                 });
        imshow(name_, show);
        return res;
    }
    int feature_;
    int curfeat_;
    int affine_;
    int curaff_;
    int limits_;
    Ptr<Feature2D> backend_;
#ifdef NON_FREE
    Ptr<AffineFeature> ext_;
#endif
};

class blob_filter : public itf_filter
{
public:
    blob_filter(const string& name) : itf_filter(name)
    {
#ifdef NON_FREE
        SimpleBlobDetector::Params pDefaultBLOB;
        pDefaultBLOB.thresholdStep = 10;
        pDefaultBLOB.minThreshold = 10;
        pDefaultBLOB.maxThreshold = 220;
        pDefaultBLOB.minRepeatability = 2;
        pDefaultBLOB.minDistBetweenBlobs = 10;
        pDefaultBLOB.filterByColor = false;
        pDefaultBLOB.blobColor = 0;
        pDefaultBLOB.filterByArea = false;
        pDefaultBLOB.minArea = 25;
        pDefaultBLOB.maxArea = 5000;
        pDefaultBLOB.filterByCircularity = false;
        pDefaultBLOB.minCircularity = 0.9f;
        pDefaultBLOB.maxCircularity = (float)1e37;
        pDefaultBLOB.filterByInertia = false;
        pDefaultBLOB.minInertiaRatio = 0.1f;
        pDefaultBLOB.maxInertiaRatio = (float)1e37;
        pDefaultBLOB.filterByConvexity = false;
        pDefaultBLOB.minConvexity = 0.95f;
        pDefaultBLOB.maxConvexity = (float)1e37;

        // Param for first BLOB detector we want all
        params_.push_back(pDefaultBLOB);
        params_.back().filterByArea = true;
        params_.back().minArea = 1;
        params_.back().maxArea = 2900;
        // Param for third BLOB detector we want only circular object
        params_.push_back(pDefaultBLOB);
        params_.back().filterByCircularity = true;
        // Param for Fourth BLOB detector we want ratio inertia
        params_.push_back(pDefaultBLOB);
        params_.back().filterByInertia = true;
        params_.back().minInertiaRatio = 0;
        params_.back().maxInertiaRatio = (float)0.2;
        // Param for fifth BLOB detector we want ratio inertia
        params_.push_back(pDefaultBLOB);
        params_.back().filterByConvexity = true;
        params_.back().minConvexity = 0.;
        params_.back().maxConvexity = (float)0.9;
        // Param for six BLOB detector we want blob with gravity center color equal to 0
        params_.push_back(pDefaultBLOB);
        params_.back().filterByColor = true;
        params_.back().blobColor = 0;
#endif
        curfeat_ = -1;
        feature_ = 0;
        affine_ = 1;
        curaff_ = affine_;
        limits_ = 0;
        update_ = 1;
        val1_ = val2_ = 500;
        area1_ = 0;
        area2_ = 2900;
        color_ = 0;
        bgorigin_ = 0;
        createTrackbar("feat count", name_, &limits_, INT_MAX, itf_filter::update_, this);
        createTrackbar("feature:\nArea:0\nCircularity:1\nInertia:2\n"
                       "Convexity:3\nColor:4\n",
                        name_, &feature_, 4, itf_filter::update_, this);
        createTrackbar("area (bound 1)", name_, &area1_, 10000, itf_filter::update_, this);
        createTrackbar("area (bound 2)", name_, &area2_, 10000, itf_filter::update_, this);
        createTrackbar("val (bound 1)", name_, &val1_, 1000, itf_filter::update_, this);
        createTrackbar("val (bound 2)", name_, &val2_, 1000, itf_filter::update_, this);
        createTrackbar("color", name_, &color_, 255, itf_filter::update_, this);
        createTrackbar("use origin (OFF/ON)", name_, &bgorigin_, 1, itf_filter::update_, this);
    }
protected:
    Ptr<Feature2D> getBackend()
    {
        if (!backends_[feature_])
        {
#ifdef NON_FREE
            backends_[feature_] = SimpleBlobDetector::create(params_[feature_]);
#endif
		}
        else if (update_)
        {
            switch (feature_)
            {
            case 0:
                {
                    SimpleBlobDetector::Params& param = params_[feature_];
                    param.minArea = min(area1_, area2_);
                    param.maxArea = max(area1_, area2_);
                }
                break;
            case 1:
                {
                    // bug if val1 >= .9, val2 should not be [.9, 1.)
                    SimpleBlobDetector::Params& param = params_[feature_];
                    param.minCircularity = .001*min(val1_, val2_);
                    param.maxCircularity = .001*max(val1_, val2_);
                }
                break;
            case 2:
                {
                    SimpleBlobDetector::Params& param = params_[feature_];
                    param.minInertiaRatio = .001*min(val1_, val2_);
                    param.maxInertiaRatio = .001*max(val1_, val2_);
                }
                break;
            case 3:
                {
                    SimpleBlobDetector::Params& param = params_[feature_];
                    param.minConvexity = .001*min(val1_, val2_);
                    param.maxConvexity = .001*max(val1_, val2_);
                }
                break;
            case 4:
                {
                    SimpleBlobDetector::Params& param = params_[feature_];
                    param.blobColor = color_;
                }
                break;
            }
#ifdef NON_FREE
            backends_[feature_] = SimpleBlobDetector::create(params_[feature_]);
#endif
		}
        return backends_[feature_];
    }
    virtual Mat _filter(Mat& image)
    {
        Mat res = image;

        bool changed = feature_ != curfeat_ || affine_ != curaff_ ;
        curaff_ = affine_;
        if (changed)
        {
           switch (feature_)
           {
           case 0:
               {
               }
               break;
           case 1:
               {
                    setTrackbarPos("val (bound 1)", name_, params_[feature_].minCircularity*1000);
                    setTrackbarPos("val (bound 2)", name_, params_[feature_].maxCircularity*1000);
               }
               break;
           case 2:
               {
                    setTrackbarPos("val (bound 1)", name_, params_[feature_].minInertiaRatio*1000);
                    setTrackbarPos("val (bound 2)", name_, params_[feature_].maxInertiaRatio*1000);
               }
               break;
           case 3:
               {
                    setTrackbarPos("val (bound 1)", name_, params_[feature_].minConvexity*1000);
                    setTrackbarPos("val (bound 2)", name_, params_[feature_].maxConvexity*1000);
               }
               break;
           case 4:
               {
               }
               break;
           }
        }
        Ptr<Feature2D> backend = getBackend();


        vector<KeyPoint> kp1;
        Mat desc1;
        Mat src = (bgorigin_)? graph_->origin() : image;
        Mat show = (bgorigin_)? graph_->origin().clone() : image.clone();
        backend->detect(image, kp1);
        setTrackbarMax("feat count", name_, kp1.size());
        setTrackbarPos("feat count", name_, kp1.size());
        if (changed)
            limits_ = kp1.size();
#ifdef NON_FREE
        drawKeypoints(src, kp1, show);
#endif
        next_color(true);
        if (limits_)
            for_each(kp1.begin(), kp1.begin() + limits_,
                     [&](KeyPoint& kp) {
                        circle(show, kp.pt, kp.size, next_color());
                     });
        imshow(name_, show);
        return res;
    }
    int feature_;
    int curfeat_;
    int affine_;
    int curaff_;
    int limits_;
    int update_;
    int area1_, area2_;
    int val1_, val2_;
    int color_;
    int bgorigin_;
    map<int, Ptr<Feature2D> > backends_;
    vector<SimpleBlobDetector::Params> params_;
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
	cut_filter(const string& name, const string& comment) : itf_filter(name, comment)
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
    virtual void _on_mouse(int event, int x, int y, int flags, void* ctx)
    {
        ostringstream os;
        switch (event)
        {
#ifdef HAVE_QT
		case EVENT_MBUTTONDBLCLK:
#else
		case EVENT_RBUTTONDBLCLK:
#endif
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
#ifdef HAVE_WIN32UI
		_wmkdir(L"pos/");
#endif
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
#ifdef HAVE_WIN32UI
		_wmkdir(L"neg/");
#endif
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
#ifdef HAVE_QT
			if (!(EVENT_FLAG_CTRLKEY & flags))
				break;
#endif
            state_ = cut_IDLE;
            rect_ = Rect();
            posrect_.clear();
            break;
#ifdef HAVE_QT
		case EVENT_MBUTTONDBLCLK:
#else
		case EVENT_RBUTTONDBLCLK:
#endif
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

typedef cut2_filter anno_filter;

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
		showconvexHull_ = 0;
		showarea_ = 0;
		last_max_area_ = 0;
		createTrackbar("threshold*.001", name_, &threshold_, 200, itf_filter::update_, this);
		createTrackbar("show poly(OFF/ON)", name_, &showpoly_, 1, itf_filter::update_, this);
		createTrackbar("show contours(OFF/ON)", name_, &showcontours_, 1, itf_filter::update_, this);
		createTrackbar("show area(OFF/MAX)", name_, &showarea_, 1, itf_filter::update_, this);
		createTrackbar("show convexHull(OFF/ON)", name_, &showconvexHull_, 1, itf_filter::update_, this);
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
		/// Z#20250303
		int area = 0;
		if (!show.empty())
		{
			area = show.rows * show.cols;
			if (area != last_max_area_)
			{
				last_max_area_ = area;
				if (showarea_ > 0)
					showarea_ = min(last_max_area_, showarea_);
				setTrackbarMax("show area(OFF/MAX)", name_, last_max_area_);
			}
		}
		else
		{
			showarea_ = 0;
			last_max_area_ = 0;
			setTrackbarMax("show area(OFF/MAX)", name_, 1);
		}

		// test each contour
		if (showpoly_)
			for (size_t i = 0; i < contours.size(); i++)
			{
				approxPolyDP(contours[i], approx, arcLength(contours[i], true)*threshold_ / 1000., true);
				polylines(show, approx, true, Scalar(255, 0, 0), 2, LINE_AA);
			}
		if (showcontours_)
			drawContours(show, contours, -1, Scalar(0, 255, 0), 1, LINE_AA);
		if (showconvexHull_)
		{
			for (size_t i = 0; i < contours.size(); i++)
			{
				std::vector<cv::Point> hull;
				auto& points = contours[i];
				cv::convexHull(points, hull);
				for (size_t i = 0; i < hull.size() && hull.size() > 5; i++)
				{
					cv::line(show, hull[i], hull[(i + 1) % hull.size()], cv::Scalar(0, 0, 255), 3, LINE_AA);
				}
			}
		}
		if (showarea_)
		{
			for (size_t i = 0; i< contours.size(); i++) 
			{
				using namespace cv;
				auto& src = show;
				double area = contourArea(contours[i]);
				double length = arcLength(contours[i], true);

				if (area < showarea_) {
					continue;
				}
				cout << "area = " << area << ", length = " << length << endl;
				RotatedRect rrt = minAreaRect(contours[i]);// 获取最小外接矩形

				Point2f pt[4];
				rrt.points(pt);
				line(src, pt[0], pt[1], Scalar(128, 128, 0), 8, 8);
				line(src, pt[1], pt[2], Scalar(128, 128, 0), 8, 8);
				line(src, pt[2], pt[3], Scalar(128, 128, 0), 8, 8);
				line(src, pt[3], pt[0], Scalar(128, 128, 0), 8, 8);
				Point  center = rrt.center;
				circle(src, center, 2, Scalar(0, 0, 128), 8, 8); // 绘制最小外接矩形的中心点
			}
		}
		imshow(name_, show);
		return image;
	}
	int threshold_;
	int retr_;
	int showpoly_;
	int showcontours_;
	int showconvexHull_;
	int showarea_;
	int last_max_area_;
};

typedef contours_filter convexHull_filter;

class match_filter : public cut_filter
{
public:
    match_filter(const string& name) : cut_filter(name, " (select) ")
    {
        use_save_ = false;
        use_mask_ = false;
        image_window_ = "Match Image " + name_;
        result_window_ = "Result window " + name_;
        namedWindow(image_window_, WINDOW_AUTOSIZE );
        namedWindow(result_window_, WINDOW_AUTOSIZE );
        createTrackbar("Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED",
                        image_window_, &match_method_, max_Trackbar_, MatchingMethod, this);
        createTrackbar("threshold*.001%", image_window_, &threshold, 10000, MatchingMethod, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        if (!apply_)
            apply([&](Mat res, Mat sel){
                  img_ = res;
                  if (!sel.empty())
                      templ_ = sel;
                  _MatchingMethod();
				  if (is_on_mouse_)
				  {
					  setWindowProperty(result_window_, 5, 0);
					  setWindowProperty(image_window_, 5, 0);
				  }
            });
        return cut_filter::_filter(image);
    }
	virtual void _on_mouse(int event, int x, int y, int flags, void* ctx)
	{
		is_on_mouse_ = true;
		cut_filter::_on_mouse(event, x, y, flags, ctx);
		is_on_mouse_ = false;
	}
    static void MatchingMethod(int, void* ctx)
    {
        ((match_filter*)ctx)->_MatchingMethod();
    }
    void _MatchingMethod()
    {
        Mat img_display;
        if (templ_.empty())
            return;
        img_.copyTo(img_display);
        if (img_.type() != templ_.type()
            && (img_.type() == CV_8UC1 && templ_.type() == CV_8UC3))
        {
            Mat tmp;
            cvtColor(templ_, tmp, CV_BGR2GRAY);
            templ_ = tmp;
        }
        int result_cols =  img_.cols - templ_.cols + 1;
        int result_rows = img_.rows - templ_.rows + 1;
        result_.create( result_rows, result_cols, CV_32FC1 );
        bool method_accepts_mask = (CV_TM_SQDIFF == match_method_ || match_method_ == CV_TM_CCORR_NORMED);
        if (use_mask_ && method_accepts_mask)
        {
            matchTemplate(img_, templ_, result_, match_method_, mask_);
        }
        else
        {
            matchTemplate(img_, templ_, result_, match_method_);
        }
        if (!(match_method_ & 1))
            normalize(result_, result_, 0, 1, NORM_MINMAX, -1, Mat());
        double minVal; double maxVal; Point minLoc; Point maxLoc;
        Point matchLoc;
        double matchVal;
        minMaxLoc(result_, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
        if (match_method_  == TM_SQDIFF || match_method_ == TM_SQDIFF_NORMED)
        {
            matchLoc = minLoc;
            matchVal = minVal;
        }
        else
        {
            matchLoc = maxLoc;
            matchVal = maxVal;
        }
        if (0 == threshold)
        {
            rectangle(img_display, matchLoc, Point(matchLoc.x + templ_.cols , matchLoc.y + templ_.rows ), Scalar::all(0), 1, LINE_AA, 0 );
            rectangle(result_, matchLoc, Point(matchLoc.x + templ_.cols , matchLoc.y + templ_.rows ), Scalar::all(0), 1, LINE_AA, 0 );
        }
        else
        {
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
            float* it = (float*)result_.data;
            const int cxr = result_.cols * result_.rows;
            double threshold1000 = threshold / 100000.;
            for (int i = 0; i < cxr; ++i, ++it)
            {
                if (fabs(*it - matchVal) <= threshold1000)
                {
                    Point pt(i % result_.cols, i / result_.cols);
                    rectangle( img_display, pt, Point(pt.x + templ_.cols , pt.y + templ_.rows), colors[i%8], 1, LINE_AA, 0 );
                    //rectangle( result, matchLoc, Point( pt.x + templ.cols , pt.y + templ.rows ), Scalar::all(0), 1, LINE_AA, 0 );
                }
            }
        }
        imshow(image_window_, img_display);
        imshow(result_window_, result_);
        return;
    }
    bool use_mask_ = false;
    Mat img_, templ_, mask_, result_;
    string image_window_;
    string result_window_;
    int match_method_ = 0;
    int max_Trackbar_ = 5;
    int threshold = 0;
    bool init = false;
	bool is_on_mouse_ = false;
};

class cascade_filter : public itf_filter
{
public:
    cascade_filter(const string& name) : itf_filter(name)
    {
        algo1_ = algo2_ = algo3_ = algo4_ = 0;
        algo2_ = 1;
        x_ = 15;
        y_ = 15;
        scalefactor_ = 15;
        minneighbros_ = 3;
        cascade_.load(samples::findFile("cascade/cascade.xml"));
        //createTrackbar("scalefactor*.01 + 1", name_, &scalefactor_, 100, itf_filter::update_, this);
        //createTrackbar("min neighbros", name_, &minneighbros_, 10, itf_filter::update_, this);
        //createTrackbar("DO_CANNY_PRUNING   (OFF/ON)", name_, &algo1_, 1, itf_filter::update_, this);
        //createTrackbar("SCALE_IMAGE        (OFF/ON)", name_, &algo2_, 1, itf_filter::update_, this);
        //createTrackbar("FIND_BIGGEST_OBJECT(OFF/ON)", name_, &algo3_, 1, itf_filter::update_, this);
        //createTrackbar("DO_ROUGH_SEARCH    (OFF/ON)", name_, &algo4_, 1, itf_filter::update_, this);
        //createTrackbar("x", name_, &x_, 100, itf_filter::update_, this);
        //createTrackbar("y", name_, &y_, 100, itf_filter::update_, this);
        createTrackbar("switch(OFF/ON)", name_, &switch_, 1, itf_filter::update_, this);
    }
protected:
    virtual Mat _filter(Mat& image)
    {
        Mat res = image;
        if (!cascade_.empty() && switch_)
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
        else
        {
            imshow(name_, res);
        }

        return res;
    }
    int switch_ = true;
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
    BRANCH(medianBlur);
    BRANCH(GaussianBlur);
    BRANCH(blur);
    BRANCH(bilateral);
    BRANCH(box);
    BRANCH(sqrBox);
    BRANCH(Sobel);
    BRANCH(Scharr);
    BRANCH(Laplacian);
    BRANCH(pyrDown);
    BRANCH(pyrUp);

    BRANCH(warpAffine);
    BRANCH(warpPerspective);
    BRANCH(warpPolar);

    BRANCH(cornerMinEigenVal);
    BRANCH(cornerHarris);
    BRANCH(cornerEigenValsAndVecs);
    BRANCH(preCornerDetect);
    BRANCH(norm_minmax);

    BRANCH(HoughLines);
    BRANCH(HoughLinesP);
    BRANCH(HoughCircles);

    BRANCH(channel);
    BRANCH(bgr2gray);
    BRANCH(gray2mask);
    BRANCH(range);
    BRANCH(colormap);
    BRANCH(cut);
    BRANCH(cut2);
    BRANCH(anno);
    BRANCH(crop);
    BRANCH(zoom);

    BRANCH(feature);
    BRANCH(blob);

    BRANCH(deskew);
    BRANCH(dem);
    BRANCH(distrans);

	BRANCH(convexHull);
    BRANCH(contours);
    BRANCH(match);
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

