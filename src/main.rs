#![cfg(feature = "cli")]
use std::path::PathBuf;
use std::process;

use opencv_yolov5::{YoloImageDetections, YoloModel};
use anyhow::Result;

use opencv::{
	prelude::*,
	imgproc,
	videoio, 
    highgui,
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
    let mut args = <Cli as clap::Parser>::parse();

    // Handle ~ in paths
    let model_path = args.model_path.canonicalize().unwrap();
    // args.root_path = args.root_path.canonicalize().unwrap();
    let video_path = args.video_path.canonicalize().unwrap();

    // model trained on 640 * 640 images.
    const CIFAR_WIDTH: i32 = 640; 
    const CIFAR_HEIGHT: i32 = 640; 
    // time that a frame will stay on screen in ms
    const DELAY: i32 = 30;

    //加载模型
    let model_progress = indicatif::ProgressBar::new_spinner();
    let mut model = YoloModel::new_from_file(
        model_path.to_str().unwrap(),
        (args.input_width, args.input_height),
    )
    .expect("Unable to load model.");
    model_progress.finish_with_message("Model loaded.");

    //存放infer结果
    let mut results: Vec<YoloImageDetections> = vec![];

    // from webcam repo:
    // create video stream 
    let mut capture = videoio::VideoCapture::from_file(video_path.to_str().unwrap(), videoio::CAP_ANY)?;
    
    println!("Inferencing on video: {}", video_path.to_str().unwrap());

    // create empty window named 'frame'
    //let win_name = "frame";
    //highgui::named_window(win_name, highgui::WINDOW_NORMAL)?;
    //highgui::resize_window(win_name, 640, 480)?;
    
    // create empty Mat to store image data
    let mut frame = Mat::default();

    // load jit model and put it to cuda
    //let mut model = tch::CModule::load(model_file)?;   
    //model.set_eval(); 
    //model.to(tch::Device::Cuda(0), tch::Kind::Float, false);

    let is_video_on = capture.is_opened()?;

    if !is_video_on {
        println!("Could'not open video. Aborting program.");
        process::exit(0);
    }
    else {
        loop {
            // read frame to empty mat 
            capture.read(&mut frame)?;
            // resize image
            let mut resized = Mat::default();   
            imgproc::resize(&frame, &mut resized, core::Size{width: CIFAR_WIDTH, height: CIFAR_HEIGHT}, 0.0, 0.0, opencv::imgproc::INTER_LINEAR)?;
            // convert bgr image to rgb
            let mut rgb_resized = Mat::default();  
            imgproc::cvt_color(&resized, &mut rgb_resized, imgproc::COLOR_BGR2RGB, 0)?;    
            // get data from Mat 
            //let h = resized.size().height();
            //let w = resized.size().width();   
            let detections = model
            .detectMat(rgb_resized, 0.1, 0.45)?;

            results.push(detections);

            // show image 
            //highgui::imshow(win_name, &frame)?;
            let key = highgui::wait_key(DELAY)?;
            // if button q pressed, abort.
            if key == 113 { 
                break;
            }
        }
    }
    
    std::fs::write(
        "output.json",
        serde_json::to_string_pretty(&results).unwrap(),
    )
    .expect("Failed to write results");
    /*
    //加载测试图像
    let images = enumerate_images(args.root_path, true);

    let image_progress = indicatif::ProgressBar::new(images.len() as u64);
    image_progress.set_style(
        indicatif::ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} {per_sec} ({eta_precise})",
            )
            .unwrap()
            .progress_chars("=> "),
    );

    
    // 执行inference
    for image_path in images {
        println!("{:?}", image_path);
        image_progress.inc(1);

        let detections = model
            .detect(image_path.to_str().unwrap(), 0.1, 0.45)
            .unwrap();

        results.push(detections);
    }

    image_progress.finish_with_message("Done.");
    */

    Ok(())
}
