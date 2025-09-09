use clap::{Parser, Subcommand};
use anyhow::Result;
use defiance_core::{DefianceNode, NodeConfig};
use defiance_core::network::NetworkEvent;
use defiance_core::content::{ContentType, ContentChunk};
use std::fs::File;
use std::io::{Read, BufReader, Write};
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;
use std::collections::HashMap;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start a broadcast session to stream a file
    Broadcast {
        /// The path to the file to stream
        #[clap(short, long)]
        file_path: String,
    },
    /// Start a viewing session to receive a stream
    View {
        /// The ID of the broadcast to view
        #[clap(short, long)]
        broadcast_id: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Broadcast { file_path } => {
            println!("Starting broadcast for file: {}", file_path);
            
            let mut node = DefianceNode::new(NodeConfig::default()).await?;
            node.start().await?;
            
            let streaming_engine = node.streaming.clone();
            let broadcast_id = {
                let mut engine = streaming_engine.write().await;
                engine.start_broadcast(
                    "CLI Broadcast".to_string(),
                    "Streaming a file from the CLI".to_string(),
                    ContentType::Video,
                ).await?
            };
            
            println!("Broadcast started with ID: {}", broadcast_id);
            
            let content_id = {
                let engine = streaming_engine.read().await;
                let broadcast = engine.get_broadcast(broadcast_id).unwrap();
                broadcast.content.metadata.id
            };
            
            let file = File::open(file_path)?;
            let mut reader = BufReader::new(file);
            let mut buffer = vec![0; defiance_core::CHUNK_SIZE];
            let mut chunk_id = 0;
            
            loop {
                let bytes_read = reader.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }
                
                let data = buffer[..bytes_read].to_vec();
                let checksum = blake3::hash(&data).into();
                
                let chunk = ContentChunk {
                    content_id,
                    chunk_id,
                    data,
                    checksum,
                    quality: defiance_core::content::Quality::Source,
                    timestamp: Some(0), // TODO: Add proper timestamp
                };
                
                {
                    let mut engine = streaming_engine.write().await;
                    engine.add_broadcast_chunk(broadcast_id, chunk).await?;
                }
                
                println!("Sent chunk {} of size {}", chunk_id, bytes_read);
                chunk_id += 1;
                sleep(Duration::from_millis(100)).await;
            }
            
            println!("Finished sending file.");
            sleep(Duration::from_secs(60)).await;
            node.stop().await?;
        }
        Commands::View { broadcast_id } => {
            println!("Starting to view broadcast: {}", broadcast_id);
            
            let mut node = DefianceNode::new(NodeConfig::default()).await?;
            node.start().await?;
            
            let broadcast_uuid = Uuid::parse_str(broadcast_id)?;
            
            // In a real application, we would discover the broadcaster's PeerId.
            // For this test, we can't join the session, but we can listen for chunks.
            // let session_id = node.join_viewing_session(broadcast_uuid, ???).await?;
            
            let mut network_events = {
                let mut network = node.network.write().await;
                network.take_event_receiver().unwrap()
            };
            
            println!("Listening for stream chunks...");
            
            loop {
                if let Ok(event) = network_events.try_recv() {
                    if let NetworkEvent::RequestResponseReceived { message, .. } = event {
                        if let libp2p::request_response::Message::Request { request, .. } = message {
                            if let defiance_core::network::NetworkMessage::StreamChunk { content_id, data, .. } = request {
                                if content_id == broadcast_uuid {
                                    // In a real app, we'd get the session_id when we join
                                    // node.video.write().await.process_packet(session_id, &data).await?;
                                    println!("Received chunk of size {}", data.len());
                                }
                            }
                        }
                    }
                }
                sleep(Duration::from_millis(100)).await;
            }
        }
    }

    Ok(())
}
