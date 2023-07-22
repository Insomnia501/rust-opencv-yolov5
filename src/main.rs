#![cfg(feature = "cli")]
use std::{path::PathBuf, process};

use opencv_yolov5::{YoloImageDetections, YoloModel};
use anyhow::Result;

use opencv::{
	prelude::*,
	imgproc,
	videoio, 
    core,
};

#[derive(clap::Parser)]
struct Cli {
    model_path: PathBuf,

    // #[clap(parse(from_os_str))]
    video_path: PathBuf,

    #[clap(long, default_value = "false")]
    recursive: bool,

    #[clap(long, default_value = "640")]
    input_width: i32,

    #[clap(long, default_value = "640")]
    input_height: i32,
}

fn main() -> Result<()> {
    println!("start process...");
    let args = <Cli as clap::Parser>::parse();

    // Handle ~ in paths
    let model_path = args.model_path.canonicalize()?;
    let video_path = args.video_path.canonicalize()?;

    // model trained on 640 * 640 images.
    const CIFAR_WIDTH: i32 = 320; 
    const CIFAR_HEIGHT: i32 = 320;

    // load model
    let model_progress = indicatif::ProgressBar::new_spinner();
    let mut model = YoloModel::new_from_file(
        model_path.to_str().unwrap(),
        (args.input_width, args.input_height),
    )
    .expect("Unable to load model.");
    model_progress.finish_with_message("Model loaded.");

    // restore inference result
    let mut results: Vec<YoloImageDetections> = vec![];

    // create video stream 
    let mut capture = videoio::VideoCapture::from_file(video_path.to_str().unwrap(), videoio::CAP_ANY)?;
    
    println!("Inferencing on video: {}", video_path.to_str().unwrap());

    // create empty Mat to store image data
    let mut frame = Mat::default();

    // sample frame
    let fps = capture.get(videoio::CAP_PROP_FPS)?;
    let frames_to_skip = (fps as i32) - 1;
    let mut frame_counter = 0;
    
    let is_video_on = capture.is_opened()?;

    if !is_video_on {
        println!("Could'not open video. Aborting program.");
        process::exit(0);
    }
    else {
        //let start = time::Instant::now();
        loop {
            let is_read = capture.read(&mut frame)?;
            if !is_read{
                println!("video end.");
                break;
            }
            // read frame to empty mat
            if frame_counter % (frames_to_skip + 1) == 0 {
                // resize image
                let mut resized = Mat::default();  
                imgproc::resize(&frame, &mut resized, core::Size{width: CIFAR_WIDTH, height: CIFAR_HEIGHT}, 0.0, 0.0, opencv::imgproc::INTER_LINEAR)?;
                // convert bgr image to rgb
                let mut rgb_resized = Mat::default();  
                imgproc::cvt_color(&resized, &mut rgb_resized, imgproc::COLOR_BGR2RGB, 0)?;    
                let detections = model
                .detect_mat(rgb_resized, 0.1, 0.45)?;
                results.push(detections);
            }
            frame_counter += 1;
        }
    }
    
    std::fs::write(
        "output.json",
        serde_json::to_string_pretty(&results)?,
    )
    .expect("Failed to write results");

    Ok(())
}
