#include "uploader_task.h"
extern "C" {
#include "log.h"
}
#include <curl/curl.h>
#include <random>
#include <cmath>

namespace {
constexpr size_t kUploadMaxBytes = 300 * 1024;
constexpr size_t kUploadQueueMaxSize = 80;
constexpr int kUploadDefaultJpegQuality = 92;
constexpr int kUploadMinLongEdge = 768;
constexpr int kUploadMaxLongEdge = 1280;
constexpr int kUploadFaceMaxLongEdge = 1440;
constexpr int kUploadPersonMaxLongEdge = 1600;

std::string generate_unique_code_12() {
    static thread_local std::mt19937 rng(std::random_device{}());
    static const char kChars[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::uniform_int_distribution<int> dist(0, (int)sizeof(kChars) - 2);

    std::string code;
    code.reserve(12);
    for (int i = 0; i < 12; ++i) {
        code.push_back(kChars[dist(rng)]);
    }
    return code;
}

void add_form_field(curl_mime* form, const char* name, const std::string& value) {
    curl_mimepart* field = curl_mime_addpart(form);
    curl_mime_name(field, name);
    curl_mime_data(field, value.c_str(), CURL_ZERO_TERMINATED);
}

cv::Mat resize_to_long_edge(const cv::Mat& img, int target_long_edge, int interpolation) {
    int current_long_edge = std::max(img.cols, img.rows);
    if (img.empty() || current_long_edge <= 0 || current_long_edge <= target_long_edge) {
        return img;
    }

    float scale = target_long_edge / static_cast<float>(current_long_edge);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(), scale, scale, interpolation);
    return resized;
}

std::vector<uchar> encode_jpeg(const cv::Mat& img, int quality) {
    std::vector<uchar> buf;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
    cv::imencode(".jpg", img, buf, params);
    return buf;
}

std::vector<uchar> encode_image_for_upload(const cv::Mat& img, int max_long_edge = kUploadMaxLongEdge) {
    if (img.empty()) {
        return {};
    }

    std::vector<uchar> original = encode_jpeg(img, kUploadDefaultJpegQuality);
    if (original.size() <= kUploadMaxBytes) {
        return original;
    }

    cv::Mat working = img;
    int original_long_edge = std::max(img.cols, img.rows);
    if (original_long_edge > max_long_edge) {
        working = resize_to_long_edge(working, max_long_edge, cv::INTER_AREA);
    }

    std::vector<uchar> best = encode_jpeg(working, kUploadDefaultJpegQuality);
    static const int kQualities[] = {90, 84, 78, 72, 66, 60, 55};

    while (true) {
        for (int quality : kQualities) {
            std::vector<uchar> compressed = encode_jpeg(working, quality);
            best = compressed;
            if (compressed.size() <= kUploadMaxBytes) {
                log_info("UploaderTask: image encoded from %zuKB to %zuKB (quality=%d, size=%dx%d)",
                         original.size() / 1024,
                         compressed.size() / 1024,
                         quality,
                         working.cols,
                         working.rows);
                return compressed;
            }
        }

        int current_long_edge = std::max(working.cols, working.rows);
        if (current_long_edge <= kUploadMinLongEdge) {
            break;
        }

        int next_long_edge = std::max(kUploadMinLongEdge, static_cast<int>(std::round(current_long_edge * 0.88f)));
        if (next_long_edge >= current_long_edge) {
            break;
        }
        working = resize_to_long_edge(working, next_long_edge, cv::INTER_AREA);
    }

    log_info("UploaderTask: image encoded from %zuKB to %zuKB (quality=%d, size=%dx%d)",
             original.size() / 1024,
             best.size() / 1024,
             55,
             working.cols,
             working.rows);
    return best;
}

double compute_laplacian_variance(const cv::Mat& gray) {
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_32F);
    cv::Scalar lap_mu, lap_sigma;
    cv::meanStdDev(lap, lap_mu, lap_sigma);
    return lap_sigma[0] * lap_sigma[0];
}

double compute_focus_measure_bgr(const cv::Mat& img) {
    if (img.empty()) {
        return 0.0;
    }
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img;
    }
    return compute_laplacian_variance(gray);
}

double compute_luma_mean_bgr(const cv::Mat& img) {
    if (img.empty()) {
        return 0.0;
    }
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img;
    }
    return cv::mean(gray)[0];
}

cv::Mat apply_unsharp_mask(const cv::Mat& src, double sigma, float amount) {
    if (src.empty()) {
        return src.clone();
    }

    cv::Mat blur;
    cv::GaussianBlur(src, blur, cv::Size(0, 0), sigma);
    cv::Mat out;
    cv::addWeighted(src, 1.0f + amount, blur, -amount, 0, out);
    return out;
}

cv::Mat compress_face_highlights(const cv::Mat& src) {
    if (src.empty()) {
        return src.clone();
    }

    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);

    cv::Mat lightness_f;
    channels[0].convertTo(lightness_f, CV_32F);
    for (int y = 0; y < lightness_f.rows; ++y) {
        float* row = lightness_f.ptr<float>(y);
        for (int x = 0; x < lightness_f.cols; ++x) {
            float value = row[x];
            if (value > 214.0f) {
                value = 214.0f + (value - 214.0f) * 0.42f;
            }
            if (value > 238.0f) {
                value = 238.0f + (value - 238.0f) * 0.22f;
            }
            row[x] = std::max(0.0f, std::min(255.0f, value));
        }
    }

    lightness_f.convertTo(channels[0], CV_8U);
    cv::merge(channels, lab);

    cv::Mat out;
    cv::cvtColor(lab, out, cv::COLOR_Lab2BGR);
    return out;
}

cv::Mat suppress_halo_artifacts(const cv::Mat& reference, const cv::Mat& enhanced) {
    if (reference.empty() || enhanced.empty() || reference.size() != enhanced.size()) {
        return enhanced.clone();
    }

    cv::Mat ref_gray, enh_gray;
    cv::cvtColor(reference, ref_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(enhanced, enh_gray, cv::COLOR_BGR2GRAY);

    cv::Mat diff;
    cv::subtract(enh_gray, ref_gray, diff);

    cv::Mat bright_mask, gain_mask, halo_mask;
    cv::threshold(ref_gray, bright_mask, 168, 255, cv::THRESH_BINARY);
    cv::threshold(diff, gain_mask, 14, 255, cv::THRESH_BINARY);
    cv::bitwise_and(bright_mask, gain_mask, halo_mask);
    cv::GaussianBlur(halo_mask, halo_mask, cv::Size(0, 0), 2.2);

    cv::Mat halo_mask_f;
    halo_mask.convertTo(halo_mask_f, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> halo_channels(3, halo_mask_f);
    cv::Mat halo3;
    cv::merge(halo_channels, halo3);

    cv::Mat ref_f, enh_f;
    reference.convertTo(ref_f, CV_32FC3);
    enhanced.convertTo(enh_f, CV_32FC3);
    cv::Mat blended = enh_f.mul(1.0f - halo3 * 0.55f) + ref_f.mul(halo3 * 0.55f);

    cv::Mat out;
    blended.convertTo(out, CV_8UC3);
    return out;
}

cv::Mat opencv_super_resolve_face(const cv::Mat& face) {
    if (face.empty()) {
        return face.clone();
    }

    int short_edge = std::min(face.cols, face.rows);
    if (short_edge <= 0) {
        return face.clone();
    }

    float target_short_edge = short_edge < 96 ? 320.0f : (short_edge < 160 ? 280.0f : 240.0f);
    float scale = std::min(2.6f, target_short_edge / static_cast<float>(short_edge));
    if (scale <= 1.04f) {
        return face.clone();
    }

    cv::Mat upscaled;
    cv::resize(face, upscaled, cv::Size(), scale, scale, scale > 1.8f ? cv::INTER_LANCZOS4 : cv::INTER_CUBIC);

    cv::Mat filtered;
    cv::bilateralFilter(upscaled, filtered, 7, 28, 18);

    cv::Mat detailed;
    cv::detailEnhance(filtered, detailed, 4.5f, 0.08f);

    cv::Mat softened;
    cv::addWeighted(filtered, 0.72, detailed, 0.28, 0, softened);
    softened = compress_face_highlights(softened);
    softened = suppress_halo_artifacts(filtered, softened);

    cv::Mat sr = apply_unsharp_mask(softened, 0.85, 0.08f);
    sr = suppress_halo_artifacts(filtered, sr);
    return sr;
}

double estimate_blur_angle_deg(const cv::Mat& gray);

double compute_gradient_energy_bgr(const cv::Mat& img) {
    if (img.empty()) {
        return 0.0;
    }

    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img;
    }

    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::Mat magnitude;
    cv::magnitude(gx, gy, magnitude);
    return cv::mean(magnitude)[0];
}

double compute_ghost_penalty_bgr(const cv::Mat& img) {
    if (img.empty()) {
        return 0.0;
    }

    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img;
    }

    double angle = estimate_blur_angle_deg(gray);
    int dx = static_cast<int>(std::round(std::cos(angle * CV_PI / 180.0) * 2.0));
    int dy = static_cast<int>(std::round(std::sin(angle * CV_PI / 180.0) * 2.0));
    if (dx == 0 && dy == 0) {
        dx = 1;
    }

    cv::Mat shifted;
    cv::Mat transform = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
    cv::warpAffine(gray, shifted, transform, gray.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    cv::Mat diff;
    cv::absdiff(gray, shifted, diff);
    double shift_diff = cv::mean(diff)[0];
    double penalty = std::max(0.0, 18.0 - shift_diff) * 2.2;

    double lap_var = compute_laplacian_variance(gray);
    if (lap_var > 160.0) {
        penalty *= 0.72;
    }
    return penalty;
}

double score_face_candidate(const cv::Mat& candidate,
                            double base_focus,
                            double base_luma,
                            double base_gradient,
                            double scale_bonus = 0.0) {
    double focus = compute_focus_measure_bgr(candidate);
    double luma = compute_luma_mean_bgr(candidate);
    double gradient = compute_gradient_energy_bgr(candidate);
    double ghost_penalty = compute_ghost_penalty_bgr(candidate) * 6.0;

    double focus_gain = (focus - base_focus) / std::max(80.0, base_focus);
    double gradient_gain = (gradient - base_gradient) / std::max(12.0, base_gradient);
    double luma_penalty = std::fabs(luma - base_luma) * 2.4;
    double oversoften_penalty =
        std::max(0.0, base_focus * 0.95 - focus) / std::max(20.0, base_focus) * 480.0;

    double score = focus_gain * 900.0 +
                   gradient_gain * 260.0 -
                   luma_penalty -
                   oversoften_penalty -
                   ghost_penalty +
                   scale_bonus * 22.0;

    if (focus >= base_focus * 1.03) {
        score += 35.0;
    }
    if (gradient >= base_gradient * 1.05) {
        score += 18.0;
    }
    return score;
}

cv::Mat upscale_small_face(const cv::Mat& face) {
    int short_edge = std::min(face.cols, face.rows);
    if (face.empty() || short_edge <= 0 || short_edge >= 240) {
        return face.clone();
    }

    float scale = std::min(1.85f, 240.0f / static_cast<float>(short_edge));
    if (scale <= 1.05f) {
        return face.clone();
    }

    cv::Mat upscaled;
    int interpolation = scale > 1.35f ? cv::INTER_LANCZOS4 : cv::INTER_CUBIC;
    cv::resize(face, upscaled, cv::Size(), scale, scale, interpolation);
    return upscaled;
}

double estimate_blur_angle_deg(const cv::Mat& gray) {
    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);

    double sum_xx = 0.0;
    double sum_yy = 0.0;
    double sum_xy = 0.0;
    for (int y = 0; y < gray.rows; ++y) {
        const float* px = gx.ptr<float>(y);
        const float* py = gy.ptr<float>(y);
        for (int x = 0; x < gray.cols; ++x) {
            double dx = px[x];
            double dy = py[x];
            sum_xx += dx * dx;
            sum_yy += dy * dy;
            sum_xy += dx * dy;
        }
    }

    double theta = 0.5 * std::atan2(2.0 * sum_xy, sum_xx - sum_yy);
    return theta * 180.0 / CV_PI;
}

cv::Mat apply_directional_unsharp(const cv::Mat& src, double angle_deg, int kernel_size, float amount) {
    if (src.empty() || kernel_size < 3 || amount <= 0.0f) {
        return src.clone();
    }

    kernel_size = std::max(3, kernel_size | 1);
    cv::Mat kernel = cv::Mat::zeros(kernel_size, kernel_size, CV_32F);
    cv::Point center(kernel_size / 2, kernel_size / 2);
    double rad = angle_deg * CV_PI / 180.0;
    cv::Point delta(
        static_cast<int>(std::round(std::cos(rad) * (kernel_size / 2))),
        static_cast<int>(std::round(std::sin(rad) * (kernel_size / 2))));
    cv::line(kernel, center - delta, center + delta, cv::Scalar(1.0f), 1, cv::LINE_AA);

    double sum_kernel = cv::sum(kernel)[0];
    if (sum_kernel <= 1e-6) {
        kernel.at<float>(center) = 1.0f;
        sum_kernel = 1.0;
    }
    kernel /= static_cast<float>(sum_kernel);

    cv::Mat src_f;
    src.convertTo(src_f, CV_32FC3);
    cv::Mat motion_blur;
    cv::filter2D(src_f, motion_blur, CV_32FC3, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
    cv::Mat sharpened = src_f + (src_f - motion_blur) * amount;
    cv::Mat out;
    sharpened.convertTo(out, CV_8UC3);
    return out;
}

cv::Mat apply_motion_wiener_restore_gray(const cv::Mat& gray, double angle_deg, int kernel_len, float snr) {
    if (gray.empty() || kernel_len < 3) {
        return gray.clone();
    }

    kernel_len = std::max(3, kernel_len | 1);
    cv::Mat psf = cv::Mat::zeros(gray.size(), CV_32F);
    cv::Point center(psf.cols / 2, psf.rows / 2);
    double rad = angle_deg * CV_PI / 180.0;
    cv::Point delta(
        static_cast<int>(std::round(std::cos(rad) * (kernel_len / 2))),
        static_cast<int>(std::round(std::sin(rad) * (kernel_len / 2))));
    cv::line(psf, center - delta, center + delta, cv::Scalar(1.0f), 1, cv::LINE_AA);
    double psf_sum = cv::sum(psf)[0];
    if (psf_sum <= 1e-6) {
        psf.at<float>(center) = 1.0f;
        psf_sum = 1.0;
    }
    psf /= static_cast<float>(psf_sum);

    cv::Mat gray_f;
    gray.convertTo(gray_f, CV_32F, 1.0 / 255.0);
    cv::Mat planes_img[] = {gray_f.clone(), cv::Mat::zeros(gray.size(), CV_32F)};
    cv::Mat complex_img;
    cv::merge(planes_img, 2, complex_img);
    cv::dft(complex_img, complex_img);

    cv::Mat planes_psf[] = {psf.clone(), cv::Mat::zeros(gray.size(), CV_32F)};
    cv::Mat complex_psf;
    cv::merge(planes_psf, 2, complex_psf);
    cv::dft(complex_psf, complex_psf);

    std::vector<cv::Mat> h_planes(2);
    cv::split(complex_psf, h_planes);
    cv::Mat mag2 = h_planes[0].mul(h_planes[0]) + h_planes[1].mul(h_planes[1]);
    cv::Mat denom = mag2 + 1.0f / std::max(0.5f, snr);

    cv::Mat real = h_planes[0] / denom;
    cv::Mat imag = -h_planes[1] / denom;
    cv::Mat inv_planes[] = {real, imag};
    cv::Mat inv_filter;
    cv::merge(inv_planes, 2, inv_filter);

    cv::mulSpectrums(complex_img, inv_filter, complex_img, 0);
    cv::idft(complex_img, gray_f, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    cv::Mat restored;
    gray_f.convertTo(restored, CV_8U, 255.0);
    return restored;
}

cv::Mat apply_motion_wiener_restore_bgr(const cv::Mat& src, double angle_deg, int kernel_len, float snr) {
    if (src.empty()) {
        return src.clone();
    }

    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Mat restored_gray = apply_motion_wiener_restore_gray(gray, angle_deg, kernel_len, snr);

    cv::Mat src_f, gray_f, restored_f;
    src.convertTo(src_f, CV_32FC3);
    gray.convertTo(gray_f, CV_32F, 1.0 / 255.0);
    restored_gray.convertTo(restored_f, CV_32F, 1.0 / 255.0);

    cv::Mat ratio;
    cv::divide(restored_f + 1e-3f, gray_f + 1e-3f, ratio);
    std::vector<cv::Mat> channels;
    cv::split(src_f, channels);
    for (auto& channel : channels) {
        channel = channel.mul(ratio);
    }
    cv::Mat restored;
    cv::merge(channels, restored);
    restored.convertTo(restored, CV_8UC3);
    return restored;
}

cv::Mat motion_deblur_enhance_face(const cv::Mat& face) {
    if (face.empty() || face.cols <= 0 || face.rows <= 0) {
        return face;
    }

    cv::Mat gray_in;
    cv::cvtColor(face, gray_in, cv::COLOR_BGR2GRAY);
    double mean_luma_before = cv::mean(gray_in)[0];
    double lap_var = compute_laplacian_variance(gray_in);
    float blur_severity = static_cast<float>(std::max(0.0, std::min(1.0, (135.0 - lap_var) / 135.0)));
    bool needs_upscale = std::min(face.cols, face.rows) < 220;
    bool needs_enhance = needs_upscale || lap_var < 210.0 || mean_luma_before < 110.0;
    if (!needs_enhance) {
        return face.clone();
    }

    cv::Mat work = upscale_small_face(face);

    if (blur_severity > 0.40f) {
        cv::Mat denoised;
        cv::bilateralFilter(work, denoised, 5, 24, 12);
        work = denoised;
    }

    if (mean_luma_before < 105.0) {
        cv::Mat lab;
        cv::cvtColor(work, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> lab_channels;
        cv::split(lab, lab_channels);
        double clip_limit = mean_luma_before < 85.0 ? 1.18 : 1.10;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip_limit, cv::Size(8, 8));
        clahe->apply(lab_channels[0], lab_channels[0]);
        cv::merge(lab_channels, lab);
        cv::cvtColor(lab, work, cv::COLOR_Lab2BGR);
    }

    cv::Mat work_f;
    work.convertTo(work_f, CV_32FC3);
    cv::Mat base_f;
    cv::GaussianBlur(work_f, base_f, cv::Size(0, 0), 0.75 + 0.20 * blur_severity);
    cv::Mat detail_f = work_f - base_f;
    float detail_gain = 0.12f + 0.08f * blur_severity;
    cv::Mat enhanced_f = work_f + detail_f * detail_gain;

    if (blur_severity > 0.35f) {
        cv::Mat gray_work;
        cv::cvtColor(work, gray_work, cv::COLOR_BGR2GRAY);
        double angle = estimate_blur_angle_deg(gray_work);
        if (blur_severity > 0.62f) {
            work = apply_motion_wiener_restore_bgr(work, angle, 13, 3.5f);
        }
        cv::Mat directional_face = apply_directional_unsharp(work, angle, blur_severity > 0.65f ? 11 : 9, 0.10f + 0.08f * blur_severity);
        directional_face.convertTo(enhanced_f, CV_32FC3);

        cv::Mat kernel = cv::getGaborKernel(cv::Size(7, 7), 1.8, angle * CV_PI / 180.0, 4.5, 0.9, 0.0, CV_32F);
        cv::Mat directional;
        cv::filter2D(gray_work, directional, CV_32F, kernel);
        cv::normalize(directional, directional, -10.0f, 10.0f, cv::NORM_MINMAX);

        std::vector<cv::Mat> channels;
        cv::split(enhanced_f, channels);
        for (auto& channel : channels) {
            channel += directional * 0.018f;
        }
        cv::merge(channels, enhanced_f);
    }

    cv::Mat enhanced;
    enhanced_f.convertTo(enhanced, CV_8UC3);
    enhanced = compress_face_highlights(enhanced);

    cv::Mat blur;
    cv::GaussianBlur(enhanced, blur, cv::Size(0, 0), 0.70 + 0.12 * blur_severity);
    cv::Mat out;
    float unsharp_amount = 0.06f + 0.03f * blur_severity;
    cv::addWeighted(enhanced, 1.0f + unsharp_amount, blur, -unsharp_amount, 0, out);
    out = suppress_halo_artifacts(work, out);

    cv::Mat gray_out;
    cv::cvtColor(out, gray_out, cv::COLOR_BGR2GRAY);
    double mean_luma_after = cv::mean(gray_out)[0];
    if (mean_luma_after > 1e-3) {
        double gain = mean_luma_before / mean_luma_after;
        gain = std::max(0.92, std::min(1.03, gain));
        out.convertTo(out, -1, gain, 0);
    }

    return out;
}

cv::Mat opencv_superres_deblur_face(const cv::Mat& face) {
    if (face.empty()) {
        return face.clone();
    }

    cv::Mat sr = opencv_super_resolve_face(face);
    cv::Mat deblurred = motion_deblur_enhance_face(sr);

    cv::Mat gray;
    cv::cvtColor(deblurred, gray, cv::COLOR_BGR2GRAY);
    double angle = estimate_blur_angle_deg(gray);
    double focus = compute_laplacian_variance(gray);
    float severity = static_cast<float>(std::max(0.0, std::min(1.0, (220.0 - focus) / 220.0)));

    if (severity > 0.28f) {
        deblurred = apply_motion_wiener_restore_bgr(deblurred, angle, severity > 0.60f ? 15 : 11, 3.8f);
    }

    cv::Mat final_face = apply_directional_unsharp(deblurred, angle, severity > 0.55f ? 11 : 9, 0.08f + 0.06f * severity);
    final_face = apply_unsharp_mask(final_face, 0.75, 0.06f);
    final_face = compress_face_highlights(final_face);
    final_face = suppress_halo_artifacts(sr, final_face);
    return final_face;
}

cv::Mat deghost_motion_face(const cv::Mat& face) {
    if (face.empty()) {
        return face.clone();
    }

    cv::Mat work = upscale_small_face(face);
    cv::Mat gray;
    cv::cvtColor(work, gray, cv::COLOR_BGR2GRAY);
    double focus = compute_laplacian_variance(gray);
    double angle = estimate_blur_angle_deg(gray);
    float severity = static_cast<float>(std::max(0.0, std::min(1.0, (235.0 - focus) / 235.0)));

    cv::Mat restored = apply_motion_wiener_restore_bgr(work, angle, severity > 0.65f ? 17 : 13, 4.6f);
    if (severity > 0.45f) {
        restored = apply_motion_wiener_restore_bgr(restored, angle, severity > 0.72f ? 13 : 11, 5.2f);
    }

    cv::Mat directional = apply_directional_unsharp(restored, angle, severity > 0.55f ? 9 : 7, 0.05f + 0.04f * severity);
    cv::Mat blended;
    cv::addWeighted(restored, 0.76, directional, 0.24, 0, blended);

    cv::Mat denoised;
    cv::bilateralFilter(blended, denoised, 5, 20, 10);
    denoised = compress_face_highlights(denoised);
    denoised = suppress_halo_artifacts(work, denoised);

    cv::Mat final_face;
    cv::addWeighted(denoised, 0.86, work, 0.14, 0, final_face);
    return final_face;
}

cv::Mat opencv_aggressive_deghost_face(const cv::Mat& face) {
    if (face.empty()) {
        return face.clone();
    }

    cv::Mat sr = opencv_super_resolve_face(face);
    cv::Mat deghosted = deghost_motion_face(sr);
    cv::Mat final_face = motion_deblur_enhance_face(deghosted);
    final_face = compress_face_highlights(final_face);
    final_face = suppress_halo_artifacts(sr, final_face);
    return final_face;
}

cv::Mat motion_deblur_enhance_person(const cv::Mat& person) {
    if (person.empty() || person.cols <= 0 || person.rows <= 0) {
        return person;
    }

    cv::Mat work = person;
    int long_edge = std::max(work.cols, work.rows);
    if (long_edge > 1440) {
        work = resize_to_long_edge(work, 1440, cv::INTER_AREA);
    }

    cv::Mat gray_in;
    cv::cvtColor(work, gray_in, cv::COLOR_BGR2GRAY);
    double mean_luma_before = cv::mean(gray_in)[0];
    double lap_var = compute_laplacian_variance(gray_in);
    float blur_severity = static_cast<float>(std::max(0.0, std::min(1.0, (180.0 - lap_var) / 180.0)));
    if (blur_severity < 0.18f && mean_luma_before >= 95.0) {
        return work.clone();
    }

    if (blur_severity > 0.35f) {
        cv::Mat denoised;
        cv::bilateralFilter(work, denoised, 5, 20, 16);
        work = denoised;
        cv::cvtColor(work, gray_in, cv::COLOR_BGR2GRAY);
    }

    cv::Mat work_f;
    work.convertTo(work_f, CV_32FC3);
    cv::Mat base_f;
    cv::GaussianBlur(work_f, base_f, cv::Size(0, 0), 0.85 + 0.30 * blur_severity);
    cv::Mat detail_f = work_f - base_f;
    float detail_gain = 0.08f + 0.06f * blur_severity;
    cv::Mat enhanced_f = work_f + detail_f * detail_gain;

    if (blur_severity > 0.38f) {
        double angle = estimate_blur_angle_deg(gray_in);
        if (blur_severity > 0.65f) {
            work = apply_motion_wiener_restore_bgr(work, angle, 15, 3.0f);
        }
        cv::Mat directional_person = apply_directional_unsharp(work, angle, blur_severity > 0.65f ? 13 : 11, 0.14f + 0.10f * blur_severity);
        directional_person.convertTo(enhanced_f, CV_32FC3);
        cv::Mat kernel = cv::getGaborKernel(cv::Size(9, 9), 2.0, angle * CV_PI / 180.0, 5.5, 0.9, 0.0, CV_32F);
        cv::Mat directional;
        cv::filter2D(gray_in, directional, CV_32F, kernel);
        cv::normalize(directional, directional, -8.0f, 8.0f, cv::NORM_MINMAX);

        std::vector<cv::Mat> channels;
        cv::split(enhanced_f, channels);
        for (auto& channel : channels) {
            channel += directional * 0.022f;
        }
        cv::merge(channels, enhanced_f);
    }

    cv::Mat enhanced;
    enhanced_f.convertTo(enhanced, CV_8UC3);
    cv::Mat blur;
    cv::GaussianBlur(enhanced, blur, cv::Size(0, 0), 0.75 + 0.16 * blur_severity);
    cv::Mat out;
    float unsharp_amount = 0.08f + 0.04f * blur_severity;
    cv::addWeighted(enhanced, 1.0f + unsharp_amount, blur, -unsharp_amount, 0, out);

    cv::Mat gray_out;
    cv::cvtColor(out, gray_out, cv::COLOR_BGR2GRAY);
    double mean_luma_after = cv::mean(gray_out)[0];
    if (mean_luma_after > 1e-3) {
        double gain = mean_luma_before / mean_luma_after;
        gain = std::max(0.94, std::min(1.04, gain));
        out.convertTo(out, -1, gain, 0);
    }

    return out;
}
}

UploaderTask::UploaderTask(const std::string& eqCode, const std::string& url)
        : eqCode(eqCode),
            serverUrl(url),
            running(false) {}

UploaderTask::~UploaderTask() { stop(); }

void UploaderTask::start() {
    if (running) return;
    running = true;
    worker = std::thread(&UploaderTask::run, this);
}

void UploaderTask::stop() {
    running = false;
    cv.notify_all();
    if (worker.joinable()) worker.join();
}

void UploaderTask::enqueue(const cv::Mat& img,
                          int cameraNumber,
                          const std::string& type,
                          const std::string& path,
                          const std::string& uniqueCode) {
    std::lock_guard<std::mutex> lock(mtx);
    if (queue.size() >= kUploadQueueMaxSize) {
        queue.pop();
        log_warn("UploaderTask: queue full(%zu), drop oldest to keep realtime path smooth", kUploadQueueMaxSize);
    }
    queue.push({img.clone(), cameraNumber, type, path, uniqueCode, 0});
    cv.notify_one();
}

void UploaderTask::enqueue(const UploadItem& item) {
    std::lock_guard<std::mutex> lock(mtx);
    if (queue.size() >= kUploadQueueMaxSize) {
        queue.pop();
        log_warn("UploaderTask: queue full(%zu), drop oldest retry item", kUploadQueueMaxSize);
    }
    queue.push(item);
    cv.notify_one();
}

void UploaderTask::setServerUrl(const std::string& url) {
    std::lock_guard<std::mutex> lock(mtx);
    serverUrl = url;
}

void UploaderTask::setEqCode(const std::string& code) {
    std::lock_guard<std::mutex> lock(mtx);
    eqCode = code;
}

void UploaderTask::setUploadSuccessCallback(UploadSuccessCallback cb) {
    std::lock_guard<std::mutex> lock(mtx);
    uploadSuccessCallback = cb;
}

void UploaderTask::run() {
    while (running) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]{ return !queue.empty() || !running; });
        if (!running && queue.empty()) break;

        UploadItem item = queue.front(); queue.pop();
        lock.unlock();

        std::string resp = uploadHttp(item.img, item.cameraNumber, item.type, item.path, item.uniqueCode);
        // 假设返回 "code":0 成功，否则重试
        if (resp != "0" && item.retry < 3) {
            item.retry++;
            log_error("upload failed, retrying (%d/3), type=%s, id=%d, path=%s",
                      item.retry, item.type.c_str(), item.cameraNumber, item.path.c_str());
            enqueue(item);
        } else if (resp != "0") {
            log_error("upload failed after max retries, dropped, type=%s, id=%d, path=%s",
                      item.type.c_str(), item.cameraNumber, item.path.c_str());
        } else {
            UploadSuccessCallback cb;
            {
                std::lock_guard<std::mutex> cbLock(mtx);
                cb = uploadSuccessCallback;
            }
            if (cb) {
                cb(item);
            }
        }
    }
}

std::string UploaderTask::uploadHttp(const cv::Mat& img,
                                     int cameraNumber,
                                     const std::string& type,
                                     const std::string& path,
                                     const std::string& uniqueCode) {
    // 使用 libcurl POST 上传 form-data
    CURL *curl = curl_easy_init();
    if (!curl) return "1";

    cv::Mat processed = img;
    if (type == "face") {
        cv::Mat classic_face = motion_deblur_enhance_face(img);
        cv::Mat sr_face = opencv_super_resolve_face(img);
        cv::Mat sr_deblur_face = opencv_superres_deblur_face(img);
        cv::Mat deghost_face = opencv_aggressive_deghost_face(img);

        double base_focus = compute_focus_measure_bgr(img);
        double base_luma = compute_luma_mean_bgr(img);
        double base_gradient = compute_gradient_energy_bgr(img);
        double classic_score = score_face_candidate(classic_face, base_focus, base_luma, base_gradient, 8.0);
        double sr_score = score_face_candidate(sr_face, base_focus, base_luma, base_gradient, 18.0);
        double sr_deblur_score = score_face_candidate(sr_deblur_face, base_focus, base_luma, base_gradient, 24.0);
        double deghost_score = score_face_candidate(deghost_face, base_focus, base_luma, base_gradient, 28.0);

        int short_edge = std::min(img.cols, img.rows);
        double blur_severity = std::max(0.0, std::min(1.0, (220.0 - base_focus) / 220.0));

        processed = classic_face;
        const char* best_mode = "classic";
        double best_score = classic_score;

        if (blur_severity > 0.72 || short_edge < 120) {
            processed = deghost_face;
            best_score = deghost_score;
            best_mode = "opencv_deghost_forced";
        } else if (blur_severity > 0.48 || short_edge < 160) {
            processed = sr_deblur_face;
            best_score = sr_deblur_score;
            best_mode = "opencv_sr_deblur_forced";
            if (deghost_score > sr_deblur_score + 18.0) {
                processed = deghost_face;
                best_score = deghost_score;
                best_mode = "opencv_deghost_preferred";
            }
        } else {
            if (sr_score > best_score) {
                processed = sr_face;
                best_score = sr_score;
                best_mode = "opencv_sr";
            }
            if (sr_deblur_score > best_score) {
                processed = sr_deblur_face;
                best_score = sr_deblur_score;
                best_mode = "opencv_sr_deblur";
            }
            if (deghost_score > best_score) {
                processed = deghost_face;
                best_score = deghost_score;
                best_mode = "opencv_deghost";
            }
        }

        log_debug("UploaderTask: OpenCV face compare short=%d blur=%.2f base_focus=%.1f classic=%.1f sr=%.1f sr_deblur=%.1f deghost=%.1f use=%s best=%.1f",
                  short_edge,
                  blur_severity,
                  base_focus,
                  classic_score,
                  sr_score,
                  sr_deblur_score,
                  deghost_score,
                  best_mode,
                  best_score);
    } else if (type == "person") {
        processed = motion_deblur_enhance_person(img);
    }

    int preferred_long_edge = kUploadMaxLongEdge;
    if (type == "face") {
        preferred_long_edge = kUploadFaceMaxLongEdge;
    } else if (type == "person") {
        preferred_long_edge = kUploadPersonMaxLongEdge;
    }
    std::vector<uchar> buf = encode_image_for_upload(processed, preferred_long_edge);
    curl_mime *form = curl_mime_init(curl);
    curl_mimepart *field;

    // image 文件
    field = curl_mime_addpart(form);
    curl_mime_name(field, "image");
    curl_mime_data(field, reinterpret_cast<char*>(buf.data()), buf.size());
    curl_mime_filename(field, "image.jpg");
    curl_mime_type(field, "image/jpeg");

    // time + camerNumber
    add_form_field(form, "time", std::to_string(time(nullptr)));
    add_form_field(form, "camerNumber", std::to_string(cameraNumber));

    // /receive/image/auto/minio 需要附加字段
    if (path.find("/receive/image/auto/minio") != std::string::npos) {
        std::string imageType = "1";
        std::string isHaveFace = "0";

        if (type == "face") {
            imageType = "2";
            isHaveFace = "1";
        }

        add_form_field(form, "imageType", imageType);
        add_form_field(form, "isHaveFace", isHaveFace);
        std::string code = uniqueCode.empty() ? generate_unique_code_12() : uniqueCode;
        add_form_field(form, "uniqueCode", code);
    }

    // eq-code header
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, ("eq-code: " + eqCode).c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    curl_easy_setopt(curl, CURLOPT_URL, (serverUrl + path).c_str());
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);

    CURLcode res = curl_easy_perform(curl);
    long response_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);

    curl_slist_free_all(headers);
    curl_mime_free(form);
    curl_easy_cleanup(curl);
    log_debug("response code %ld, curl result %d", response_code, res);
    return (res == CURLE_OK && response_code == 200) ? "0" : "1";
}
