#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>

using namespace std;

using namespace cv;

const int dims = 3; // number of dimensions/coordinates/features for each point
int num_points = 50; // number of points in the image, set later in getCubeVertices()
const int new_dim = 2, old_dim = 3;
const int scale = 100;

void getCubeVertices(vector<Point3d> &cube_vertices) {
    num_points = 8;
    cube_vertices.push_back(Point3d(-1,-1,-1));
    cube_vertices.push_back(Point3d(1,-1,-1));
    cube_vertices.push_back(Point3d(1,1,-1));
    cube_vertices.push_back(Point3d(-1,1,-1));
    cube_vertices.push_back(Point3d(-1,-1,1));
    cube_vertices.push_back(Point3d(1,-1,1));
    cube_vertices.push_back(Point3d(1,1,1));
    cube_vertices.push_back(Point3d(-1,1,1));
}

void multiplyMatrices(vector<double> &projection, vector<Point3d> cv, int index, vector<Point2d> &proj_v) { // matrix, vector, result
    double total = 0;
    Point3d pt = cv[index];
    vector<double> v, p{pt.x, pt.y, pt.z};
    for(int i = 0; i < new_dim; i++) {
        for(int j = 0; j < old_dim; j++) {
            total += (projection[i*old_dim + j] * p[j]);
        }
        v.push_back(total);
        total = 0;
    }
    Point2d new_pt = Point2d(v[0], v[1]);
    if(proj_v.size() < cv.size()) {
        proj_v.push_back(new_pt);
    }
    else {
        proj_v[index] = new_pt;
    }
}

void multiplyMatrices(vector<double> &rotation, vector<Point3d> &cv, int index, vector<Point3d> &rot_v) {
    double total = 0;
    Point3d pt = cv[index];
    vector<double> v, p{pt.x, pt.y, pt.z};
    for(int i = 0; i < old_dim; i++) {
        for(int j = 0; j < old_dim; j++) {
            total += (rotation[i*old_dim + j] * p[j]);
        }
        v.push_back(total);
        total = 0;
    }
    Point3d new_pt = Point3d(v[0], v[1], v[2]);
    if(rot_v.size() < cv.size()) {
        rot_v.push_back(new_pt);
    }
    else {
        rot_v[index] = new_pt;
    }
}

void extrinsicRotation(vector<double> &rotx, vector<double> &roty, vector<double> &rotz, vector<Point3d> &cv, vector<Point3d> &rv, int i) {
    multiplyMatrices(rotz, cv, i, rv);
    multiplyMatrices(roty, rv, i, rv);
    multiplyMatrices(rotx, rv, i, rv);
    // intrinsic rotation
    /* multiplyMatrices(rotx, cv, i, rv);
    multiplyMatrices(roty, rv, i, rv);
    multiplyMatrices(rotz, rv, i, rv); */
}

void intrinsicRotation(vector<double> &rotx, vector<double> &roty, vector<double> &rotz, vector<Point3d> &cv, vector<Point3d> &rv, int i) {
    multiplyMatrices(rotx, cv, i, rv);
    multiplyMatrices(roty, rv, i, rv);
    multiplyMatrices(rotz, rv, i, rv);
}

void orthographicProjection(vector<Point3d> &rot_v, int index, vector<Point3d> &proj_v) {
    vector<double> projection{1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0};
    multiplyMatrices(projection, rot_v, index, proj_v);
}

void perspectiveProjection(vector<Point3d> &cv, int index, vector<Point2d> &proj_v, Point3d &cam) {
    Point3d point = cv[index];
    Point3d ptonline{5.0,0.0,0.0};
    double t = (ptonline.x - cam.x) / (point.x - cam.x);
    ptonline.y = cam.y*(1-t) + point.y*t;
    ptonline.z = cam.z*(1-t) + point.z*t;
    Point2d new_pt = Point2d(ptonline.y, ptonline.z);
    if(proj_v.size() < cv.size()) {
        proj_v.push_back(new_pt);
    }
    else {
        proj_v[index] = new_pt;
    }
}

void drawCube(vector<Point2d> &pv, Mat &img) {
    vector<Point2d> ctrs;
    for(int i = 0; i < pv.size(); i++) {
        Point2d center = Point((scale*2)+(scale*pv[i].x), (scale*2)+(scale*pv[i].y));
        circle(img, center, 0, Scalar(255,255,255), -1);
        ctrs.push_back(center);
    }
    for(int i = 0; i < 4; i++) {
        line(img, ctrs[i], ctrs[(i+1)%4], Scalar(255,255,255), 1, LINE_AA);
        line(img, ctrs[i+4], ctrs[((i+1)%4)+4], Scalar(255,255,255), 1, LINE_AA);
        line(img, ctrs[i], ctrs[i+4], Scalar(255,255,255), 1, LINE_AA);
    }
}

int main()
{
    vector<Point3d> cube_vertices, rotated_vertices;
    vector<Point2d> projected_vertices;
    Point3d camera{10.0,0.0,0.0};
    getCubeVertices(cube_vertices);
    vector<Mat> images;
    for(int theta = 0; theta < 360; theta += 2.5) {
        vector<double> rotatex{1.0, 0.0, 0.0,
                               0.0, cos(theta*M_PI/180), -1*sin(theta*M_PI/180),
                               0.0, sin(theta*M_PI/180), cos(theta*M_PI/180)};
        vector<double> rotatey{cos(theta*M_PI/180), 0.0, -1*sin(theta*M_PI/180),
                               0.0, 1.0, 0.0,
                               sin(theta*M_PI/180), 0.0, cos(theta*M_PI/180)};
        vector<double> rotatez{cos(theta*M_PI/180), -1*sin(theta*M_PI/180), 0.0,
                               sin(theta*M_PI/180), cos(theta*M_PI/180), 0.0,
                               0.0, 0.0, 1.0};
        for(int i = 0; i < cube_vertices.size(); i++) {
            extrinsicRotation(rotatex, rotatey, rotatez, cube_vertices, rotated_vertices, i);
            // orthographicProjection(rotated_vertices, i, projected_vertices);
            perspectiveProjection(rotated_vertices, i, projected_vertices, camera);
        }
        Mat image(scale * 4, scale * 4, CV_8UC3, cv::Scalar(0, 0, 0));
        drawCube(projected_vertices, image);
        images.push_back(image);
    }
    /*for(int yaw = 0; yaw < 360; yaw += 2.5) {
        int pitch = yaw, roll = yaw;
        vector<double> rotatelocalx{1.0, 0.0, 0.0,
                                    0.0, cos(roll*M_PI/180), -1*sin(roll*M_PI/180),
                                    0.0, sin(roll*M_PI/180), cos(roll*M_PI/180)};
        vector<double> rotatelocaly{cos(pitch*M_PI/180), 0.0, -1*sin(pitch*M_PI/180),
                               0.0, 1.0, 0.0,
                               sin(pitch*M_PI/180), 0.0, cos(pitch*M_PI/180)};
        vector<double> rotatelocalz{cos(yaw*M_PI/180), -1*sin(yaw*M_PI/180), 0.0,
                               sin(yaw*M_PI/180), cos(yaw*M_PI/180), 0.0,
                               0.0, 0.0, 1.0};
        for(int i = 0; i < cube_vertices.size(); i++) {
            intrinsicRotation(rotatelocalx, rotatelocaly, rotatelocalz, cube_vertices, rotated_vertices, i);
            // orthographicProjection(rotated_vertices, i, projected_vertices);
            perspectiveProjection(rotated_vertices, i, projected_vertices, camera);
        }
        Mat image(scale * 4, scale * 4, CV_8UC3, cv::Scalar(0, 0, 0));
        drawCube(projected_vertices, image);
        images.push_back(image);
    }*/
    // imwrite("./image.jpg", images[0]);
    
    Size S = images[0].size();
    
    VideoWriter outputVideo;
    bool isColor = (images[0].type() == CV_8UC3);
    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');  // select desired codec (must be available at runtime)
    double fps = 8.0;                          // framerate of the created video stream
    string filename = "./rotation.avi";             // name of the output video file
    outputVideo.open(filename, codec, fps, S, isColor);
    
    if (!outputVideo.isOpened()){
        cout  << "Could not open the output video for write: "<< endl;
        return -1;
    }

    for(int i=0; i<images.size(); i++){
        outputVideo.write(images[i]);
    }
    
    waitKey(0);
    return 0;
}