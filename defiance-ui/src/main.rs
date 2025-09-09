use eframe::{egui, App, Frame};
use egui::Context;
use defiance_core::{DefianceNode, NodeConfig};
use defiance_core::content::{ContentType, ContentChunk};
use std::sync::mpsc::{channel, Sender, Receiver};
use tokio::runtime::Runtime;
use std::fs::File;
use std::io::{Read, BufReader};
use std::time::Duration;
use tokio::time::sleep;

enum UiMessage {
    StartBroadcast(String),
    JoinBroadcast(String),
    GetBroadcasts,
}

enum NodeMessage {
    BroadcastList(Vec<Broadcast>),
    Log(String),
}

#[derive(PartialEq)]
enum CurrentPage {
    Home,
    Browse,
    MyLibrary,
    Settings,
}

#[derive(Clone)]
struct Broadcast {
    id: String,
    title: String,
    description: String,
    caster: String,
}

struct DefianceApp {
    ui_sender: Sender<UiMessage>,
    node_receiver: Receiver<NodeMessage>,
    messages: Vec<String>,
    current_page: CurrentPage,
    broadcasts: Vec<Broadcast>,
    stream_to_watch: String,
}

impl Default for DefianceApp {
    fn default() -> Self {
        let (ui_sender, node_ui_receiver) = channel();
        let (node_sender, ui_receiver) = channel();

        std::thread::spawn(move || {
            let rt = Runtime::new().unwrap();
            rt.block_on(async {
                let mut node = DefianceNode::new(NodeConfig::default()).await.unwrap();
                node.start().await.unwrap();

                let mut network_events = {
                    let mut network = node.network.write().await;
                    network.take_event_receiver().unwrap()
                };

                loop {
                    if let Ok(event) = network_events.try_recv() {
                        if let defiance_core::network::NetworkEvent::BroadcastReceived { message, .. } = event {
                            if let Ok(announcement) = bincode::deserialize::<defiance_core::network::NetworkMessage>(&message) {
                                if let defiance_core::network::NetworkMessage::BroadcastAnnouncement { broadcast_id, title, description, broadcaster, .. } = announcement {
                                    let mut broadcasts = node.active_broadcasts.write().await;
                                    // This is incorrect, we should be creating a BroadcastSession here.
                                    // For now, we'll just log it.
                                    println!("Received broadcast announcement: {} - {}", title, broadcaster);
                                }
                            }
                        }
                    }
                
                    if let Ok(msg) = node_ui_receiver.try_recv() {
                        match msg {
                            UiMessage::StartBroadcast(file_path) => {
                                println!("Node thread received start broadcast for: {}", file_path);
                                
                                let streaming_engine = node.streaming.clone();
                                let broadcast_id = {
                                    let mut engine = streaming_engine.write().await;
                                    engine.start_broadcast(
                                        "UI Broadcast".to_string(),
                                        "Streaming from the UI".to_string(),
                                        ContentType::Video,
                                    ).await.unwrap()
                                };
                                
                                node_sender.send(NodeMessage::Log(format!("Broadcast started with ID: {}", broadcast_id))).unwrap();
                                
                                let file = File::open(file_path).unwrap();
                                let mut reader = BufReader::new(file);
                                let mut buffer = vec![0; defiance_core::CHUNK_SIZE];
                                let mut chunk_id = 0;
                                
                                loop {
                                    let bytes_read = reader.read(&mut buffer).unwrap();
                                    if bytes_read == 0 {
                                        break;
                                    }
                                    
                                    let data = buffer[..bytes_read].to_vec();
                                    let checksum = blake3::hash(&data).into();
                                    
                                    let chunk = ContentChunk {
                                        content_id: broadcast_id,
                                        chunk_id,
                                        data,
                                        checksum,
                                        quality: defiance_core::content::Quality::Source,
                                        timestamp: Some(0),
                                    };
                                    
                                    {
                                        let mut engine = streaming_engine.write().await;
                                        engine.add_broadcast_chunk(broadcast_id, chunk).await.unwrap();
                                    }
                                    
                                    node_sender.send(NodeMessage::Log(format!("Sent chunk {} of size {}", chunk_id, bytes_read))).unwrap();
                                    chunk_id += 1;
                                    sleep(Duration::from_millis(100)).await;
                                }
                                
                                node_sender.send(NodeMessage::Log("Finished sending file.".to_string())).unwrap();
                            }
                            UiMessage::JoinBroadcast(broadcast_id) => {
                                println!("Node thread received join broadcast for: {}", broadcast_id);
                                let broadcast_uuid = uuid::Uuid::parse_str(&broadcast_id).unwrap();
                                // TODO: Find the broadcaster's PeerId
                                // TODO: Send a JoinBroadcast message
                            }
                            UiMessage::GetBroadcasts => {
                                let broadcasts = node.active_broadcasts.read().await;
                                let broadcast_list = broadcasts.values().map(|b| Broadcast {
                                    id: b.id.to_string(),
                                    title: b.content.metadata.title.clone(),
                                    description: b.content.metadata.description.clone(),
                                    caster: b.content.metadata.creator.clone(),
                                }).collect();
                                node_sender.send(NodeMessage::BroadcastList(broadcast_list)).unwrap();
                            }
                        }
                    }
                }
            });
        });

        Self {
            ui_sender,
            node_receiver: ui_receiver,
            messages: Vec::new(),
            current_page: CurrentPage::Home,
            broadcasts: vec![
                Broadcast {
                    id: "1".to_string(),
                    title: "My First Stream".to_string(),
                    description: "This is a test stream.".to_string(),
                    caster: "Alice".to_string(),
                },
                Broadcast {
                    id: "2".to_string(),
                    title: "Cooking with Bob".to_string(),
                    description: "Join me as I cook a delicious meal.".to_string(),
                    caster: "Bob".to_string(),
                },
                Broadcast {
                    id: "3".to_string(),
                    title: "Gaming Live".to_string(),
                    description: "Playing the latest and greatest games.".to_string(),
                    caster: "Charlie".to_string(),
                },
            ],
            stream_to_watch: String::new(),
        }
    }
}

impl App for DefianceApp {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        if let Ok(msg) = self.node_receiver.try_recv() {
            match msg {
                NodeMessage::Log(log) => self.messages.push(log),
                NodeMessage::BroadcastList(broadcasts) => self.broadcasts = broadcasts,
            }
        }
    
        egui::SidePanel::left("sidebar").show(ctx, |ui| {
            ui.heading("Defiance Network");
            ui.separator();
            if ui.button("Home").clicked() {
                self.current_page = CurrentPage::Home;
            }
            if ui.button("Browse").clicked() {
                self.current_page = CurrentPage::Browse;
                self.ui_sender.send(UiMessage::GetBroadcasts).unwrap();
            }
            if ui.button("My Library").clicked() {
                self.current_page = CurrentPage::MyLibrary;
            }
            ui.separator();
            if ui.button("Settings").clicked() {
                self.current_page = CurrentPage::Settings;
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.current_page {
                CurrentPage::Home => {
                    ui.heading("Home");
                    ui.separator();
                    ui.label("Welcome to Defiance Network!");
                }
                CurrentPage::Browse => {
                    ui.heading("Browse Broadcasts");
                    ui.separator();
                    for broadcast in &self.broadcasts {
                        ui.group(|ui| {
                            ui.heading(&broadcast.title);
                            ui.label(&broadcast.description);
                            ui.label(format!("Caster: {}", broadcast.caster));
                            if ui.button("Watch").clicked() {
                                self.stream_to_watch = broadcast.id.clone();
                                self.ui_sender.send(UiMessage::JoinBroadcast(broadcast.id.clone())).unwrap();
                            }
                        });
                    }
                }
                CurrentPage::MyLibrary => {
                    ui.heading("My Library");
                    ui.separator();
                    ui.label("This is where your saved content will go.");
                }
                CurrentPage::Settings => {
                    ui.heading("Settings");
                    ui.separator();
                    ui.label("This is where your settings will go.");
                }
            }
        });
        
        ctx.request_repaint(); // Ensure the UI keeps updating
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Defiance Network",
        options,
        Box::new(|_cc| Ok(Box::new(DefianceApp::default()))),
    )
}
