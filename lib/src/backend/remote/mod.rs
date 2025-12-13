pub mod server;
pub mod client;
pub mod protocol;

pub(crate) type TensorId = u32;


#[cfg(test)]
mod tests {
    use std::net::IpAddr;

    use crate::backend::{remote::{self, client::RemoteBackend, server::BackendServer}, Backend};

    #[test]
    fn my_debug_space() {
        // Initialize tracing subscriber for debug logs
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_target(true)
            .with_thread_ids(true)
            .init();

        tracing::info!("Starting my_debug_space test");
        
        let server_port = 8080;
        let server_address = IpAddr::V4("127.0.0.1".parse().unwrap());
        
        tracing::debug!("Creating BackendServer on {}:{}", server_address, server_port);
        let mut remote_server = BackendServer::new(server_address, server_port);
        
        tracing::debug!("Spawning server thread");
        let server_thread = std::thread::spawn(move || {
            tracing::debug!("Server thread started, calling serve()");
            remote_server.serve().unwrap();
            tracing::debug!("Server thread ending");
        });
        
        tracing::debug!("Sleeping for 100ms to let server start");
        std::thread::sleep(std::time::Duration::from_millis(100));
        
        tracing::debug!("Creating RemoteBackend client");
        let mut backend = RemoteBackend::new(server_address, server_port);
        
        tracing::debug!("Attempting to connect to remote backend");
        backend.connect().unwrap();
        println!("Connected to remote backend");
        tracing::info!("Successfully connected to remote backend");
        
        tracing::debug!("Allocating buffer of 10 u8 elements");
        let mut buffer = backend.alloc::<u8>(10).unwrap();
        tracing::info!("Successfully allocated buffer");
        backend.write(&mut buffer, 0, 42).unwrap();
        

        // read
        let value: u8 = backend.read(&buffer, 0).unwrap();
        tracing::info!("Read value from remote backend: {}", value);
        assert_eq!(value, 42);

        tracing::debug!("Joining server thread");
        server_thread.join().unwrap();
        tracing::info!("Test completed successfully");
    }
}