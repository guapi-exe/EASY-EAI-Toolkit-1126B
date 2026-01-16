#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <sys/time.h>
#include "face_detect_retian.h"

using namespace cv;
using namespace std;

// Color array for visualization
static Scalar colorArray[10] = {
    Scalar(255, 0, 0, 255),
    Scalar(0, 255, 0, 255),
    Scalar(0, 0, 139, 255),
    Scalar(0, 100, 0, 255),
    Scalar(139, 139, 0, 255),
    Scalar(209, 206, 0, 255),
    Scalar(0, 127, 255, 255),
    Scalar(139, 61, 72, 255),
    Scalar(0, 255, 0, 255),
    Scalar(255, 0, 0, 255),
};

// Draw detection results on image
void draw_results(Mat &img, const vector<RetinaFaceResult> &results) {
    for (size_t i = 0; i < results.size(); i++) {
        const RetinaFaceResult &face = results[i];
        
        // Draw bounding box
        int x1 = (int)face.box.x;
        int y1 = (int)face.box.y;
        int x2 = (int)face.box.br().x;
        int y2 = (int)face.box.br().y;
        
        rectangle(img, Point(x1, y1), Point(x2, y2), colorArray[i % 10], 2);
        
        // Draw confidence score
        char text[64];
        sprintf(text, "Face %.2f%%", face.score * 100);
        int baseline = 0;
        Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        rectangle(img, Point(x1, y1 - textSize.height - 5), 
                  Point(x1 + textSize.width, y1), colorArray[i % 10], -1);
        putText(img, text, Point(x1, y1 - 3), FONT_HERSHEY_SIMPLEX, 
                0.5, Scalar(255, 255, 255), 1);
        
        // Draw landmarks
        for (size_t j = 0; j < face.landmarks.size(); j++) {
            circle(img, Point((int)face.landmarks[j].x, (int)face.landmarks[j].y), 
                   3, Scalar(0, 255, 255), -1);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <model_path> [image_path] [model_type]\n", argv[0]);
        printf("  model_type: 0=RETINAFACE (default), 1=SLIM, 2=RFB\n");
        printf("Example:\n");
        printf("  %s ./retinaface_480x640.rknn test.jpg 0\n", argv[0]);
        return -1;
    }
    
    const char *model_path = argv[1];
    const char *image_path = (argc > 2) ? argv[2] : "test.jpg";
    int model_type_int = (argc > 3) ? atoi(argv[3]) : 0;
    
    RetinaFaceModelType model_type = RETINAFACE_MODEL;
    if (model_type_int == 1) model_type = SLIM_MODEL;
    else if (model_type_int == 2) model_type = RFB_MODEL;
    
    // Load image
    Mat img = imread(image_path);
    if (img.empty()) {
        printf("Failed to load image: %s\n", image_path);
        return -1;
    }
    printf("Image loaded: %dx%d\n", img.cols, img.rows);
    
    // Get model configuration (input size: 480x640)
    RetinaFaceConfig config = get_retian_config(model_type, 480, 640);
    
    // Initialize model
    rknn_context ctx;
    if (face_detect_retian_init(&ctx, model_path, &config) != 0) {
        printf("Failed to initialize RetinaFace model\n");
        return -1;
    }
    
    // Run detection
    vector<RetinaFaceResult> results;
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    int num_faces = face_detect_retian_run(ctx, img, results, 0.5f, 0.4f, 10);
    
    gettimeofday(&end, NULL);
    float time_use = (end.tv_sec - start.tv_sec) * 1000.0 + 
                     (end.tv_usec - start.tv_usec) / 1000.0;
    
    printf("\n=== Detection Results ===\n");
    printf("Inference time: %.2f ms\n", time_use);
    printf("Detected %d face(s)\n\n", num_faces);
    
    // Print results
    for (int i = 0; i < num_faces; i++) {
        printf("Face %d:\n", i + 1);
        results[i].print();
        printf("\n");
    }
    
    // Draw and save results
    draw_results(img, results);
    
    const char *output_path = "result.jpg";
    imwrite(output_path, img);
    printf("Result saved to: %s\n", output_path);
    
    // Release model
    face_detect_retian_release(ctx);
    
    return 0;
}
