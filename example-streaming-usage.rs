//! Example usage of DefianceNetwork video streaming and casting features

use defiance_core::{
    DefianceNode, NodeConfig, 
    video::{VideoEngine, VideoEngineConfig, VideoQuality, VideoResolution, StreamCategory},
    User, Username
};
use defiance_cast::{CastingManager, CastingConfig, MediaMetadata};
use uuid::Uuid;
use url::Url;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::init();

    println!("🎬 DefianceNetwork Video Streaming & Casting Demo");
    println!("==================================================");

    // Example 1: Create and start a video stream
    demo_video_streaming().await?;
    
    // Example 2: Demonstrate casting to devices
    demo_casting().await?;

    Ok(())
}

/// Demonstrate video streaming functionality
async fn demo_video_streaming() -> Result<()> {
    println!("\n🎥 Video Streaming Demo");
    println!("-----------------------");

    // Create DefianceNetwork node
    let config = NodeConfig::default();
    let mut node = DefianceNode::new(config).await?;
    node.start().await?;

    // Create a mock broadcaster
    let broadcaster = User {
        id: Uuid::new_v4(),
        username: Username::generate(),
        created_at: chrono::Utc::now().timestamp(),
        last_seen: chrono::Utc::now().timestamp(),
        opt_in_visibility: true,
        reputation_score: 4.5,
        total_broadcasts: 15,
        total_watch_time: 0,
    };

    println!("👤 Broadcaster: {}", broadcaster.username.value);

    // Create video engine
    let video_config = VideoEngineConfig {
        max_concurrent_streams: 10,
        enable_recording: true,
        adaptive_quality: true,
        ..Default::default()
    };
    let mut video_engine = VideoEngine::new(video_config).await?;
    video_engine.start().await?;

    // Create a new stream
    let stream_id = video_engine.create_stream(
        broadcaster.clone(),
        "DefianceNetwork Tech Talk".to_string(),
        "Live discussion about decentralized streaming technology".to_string(),
        StreamCategory::Technology,
        VideoResolution::new_1080p(),
        30.0, // 30 FPS
    ).await?;

    println!("📡 Created stream: {}", stream_id);

    // Start the stream
    video_engine.start_stream(stream_id).await?;
    println!("🔴 Stream is now LIVE!");

    // Simulate viewers joining
    for i in 1..=3 {
        let viewer_id = Uuid::new_v4();
        let session_id = video_engine.connect_viewer(viewer_id, stream_id).await?;
        println!("👁️  Viewer {} joined (session: {})", i, session_id);
        
        // Change quality for second viewer
        if i == 2 {
            video_engine.change_viewer_quality(session_id, VideoQuality::Medium).await?;
            println!("📺 Viewer {} switched to Medium quality", i);
        }
    }

    // Get stream info
    if let Some(stream) = video_engine.get_stream(stream_id).await {
        println!("📊 Stream stats:");
        println!("   - Resolution: {}x{}", stream.resolution.width, stream.resolution.height);
        println!("   - Bitrate: {} bps", stream.bitrate);
        println!("   - Viewers: {}", stream.current_viewers.len());
        println!("   - State: {:?}", stream.state);
    }

    // Simulate some streaming time
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Stop the stream
    video_engine.stop_stream(stream_id).await?;
    println!("⏹️  Stream ended");

    video_engine.stop().await?;
    node.stop().await?;

    Ok(())
}

/// Demonstrate casting functionality
async fn demo_casting() -> Result<()> {
    println!("\n📱 Casting Demo");
    println!("----------------");

    // Create casting manager
    let cast_config = CastingConfig {
        enable_chromecast: true,
        enable_airplay: true,
        enable_dlna: true,
        auto_discovery: true,
        ..Default::default()
    };

    let mut casting_manager = CastingManager::with_config(cast_config).await?;
    casting_manager.start().await?;

    // Wait for device discovery
    println!("🔍 Discovering casting devices...");
    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    // Get available devices
    let devices = casting_manager.get_all_devices().await;
    println!("📱 Found {} casting devices:", devices.len());
    
    for device in &devices {
        println!("   - {} ({:?}) - {:?}", 
            device.name, 
            device.device_type, 
            device.status
        );
    }

    if !devices.is_empty() {
        let device = &devices[0];
        println!("\n🎯 Casting to: {}", device.name);

        // Cast some content
        let content_url = Url::parse("https://cdn.defiancenetwork.org/demo/sample-video.mp4")?;
        
        let session_id = casting_manager.cast_content(
            device.id.clone(),
            content_url,
            "DefianceNetwork Demo Video".to_string(),
            "video/mp4".to_string(),
            Uuid::new_v4(), // user_id
        ).await?;

        println!("✅ Started casting session: {}", session_id);

        // Simulate some playback time
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        println!("⏹️  Ending casting session");
        // Note: Session would be ended automatically when casting_manager is dropped
    } else {
        println!("❌ No casting devices found (this is expected in demo mode)");
    }

    // Get casting stats
    let stats = casting_manager.get_stats().await;
    println!("\n📊 Casting Statistics:");
    println!("   - Total devices: {}", stats.total_devices);
    println!("   - Available: {}", stats.available_devices);
    println!("   - Busy: {}", stats.busy_devices);
    println!("   - Enabled protocols: {:?}", stats.protocols_enabled);

    casting_manager.stop().await?;

    Ok(())
}

/// Example of creating custom media metadata
fn create_sample_metadata() -> MediaMetadata {
    MediaMetadata::new_video(
        "DefianceNetwork: The Future of Streaming".to_string(),
        Some(1800.0) // 30 minutes
    )
    .with_description("Learn about decentralized streaming technology and how DefianceNetwork is revolutionizing content distribution.".to_string())
    .with_thumbnail(
        Url::parse("https://cdn.defiancenetwork.org/thumbnails/future-streaming.jpg").unwrap()
    )
}

/// Example of handling different video qualities
fn demonstrate_quality_selection() {
    println!("\n🎞️  Video Quality Options:");
    
    let qualities = vec![
        VideoQuality::Low,
        VideoQuality::Medium, 
        VideoQuality::High,
        VideoQuality::Ultra,
        VideoQuality::Auto,
    ];

    for quality in qualities {
        let resolution = quality.get_resolution();
        let bitrate = quality.get_bitrate();
        
        println!("   - {:?}: {}x{} @ {} bps", 
            quality, 
            resolution.width, 
            resolution.height, 
            bitrate
        );
    }
}

/// Example of Renaissance/eco-friendly UI concepts
fn demonstrate_ui_concepts() {
    println!("\n🌿 Renaissance/Eco-Friendly UI Concepts:");
    println!("   - Color Palette: Earth tones, forest greens, warm golds");
    println!("   - Typography: Serif headers, clean sans-serif body");
    println!("   - Imagery: Tree motifs, water elements, renewable energy");
    println!("   - Design: Organic shapes, flowing lines, nature-inspired");
    println!("   - Themes: Both dark and light modes with sustainable aesthetic");
}