import multiprocessing as mp
import queue
import time
import logging
import os
import signal
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional

class ChainStatus(Enum):
    INITIALIZING = auto()
    RUNNING = auto()
    COLLECTING = auto()
    COMPLETE = auto()
    ERROR = auto()

class MessageType(Enum):
    SAMPLE = auto()
    START_COLLECTION = auto()
    COLLECTION_COMPLETE = auto()
    ERROR = auto()
    HEARTBEAT = auto()

@dataclass
class Message:
    def __init__(self, type: MessageType, chain_id: int, 
                 data: Optional[Dict] = None, 
                 error: Optional[str] = None,
                 iteration: Optional[int] = None):
        self.type = type
        self.chain_id = chain_id
        self.data = data
        self.error = error
        self.iteration = iteration

class ProcessManager:
    def __init__(self, num_chains: int, timeout: float = 600.0):
        self.num_chains = num_chains
        self.timeout = timeout
        self.chain_statuses = {i: ChainStatus.INITIALIZING for i in range(num_chains)}
        self.last_heartbeats = {i: time.time() for i in range(num_chains)}
        self.processes: List[mp.Process] = []
        
        # Initialize queues
        self.monitor_queues = [mp.Queue() for _ in range(num_chains)]
        self.result_queues = [mp.Queue() for _ in range(num_chains)]
        self.control_queues = [mp.Queue() for _ in range(num_chains)]
        self.monitor_queue = mp.Queue()
        
    def setup_queues(self):
        """Initialize all communication queues."""
        self.monitor_queues = [mp.Queue() for _ in range(self.num_chains)]
        self.result_queues = [mp.Queue() for _ in range(self.num_chains)]
        self.control_queues = [mp.Queue() for _ in range(self.num_chains)]
        
    def cleanup_queues(self):
        """Safely clean up all queues."""
        for q in self.monitor_queues + self.result_queues + self.control_queues + [self.monitor_queue]:
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error cleaning queue: {str(e)}")
                
    def terminate_processes(self):
        """Safely terminate all processes."""
        for p in self.processes:
            try:
                p.terminate()
                p.join(timeout=1.0)
                if p.is_alive():
                    os.kill(p.pid, signal.SIGTERM)
            except Exception as e:
                logging.error(f"Error terminating process: {str(e)}")
                
    def cleanup(self):
        """Perform complete cleanup of all resources."""
        self.terminate_processes()
        self.cleanup_queues()
        
    def check_heartbeats(self) -> bool:
        """Check if any chains have timed out."""
        current_time = time.time()
        for chain_id, last_time in self.last_heartbeats.items():
            if (current_time - last_time > self.timeout and 
                self.chain_statuses[chain_id] not in [ChainStatus.COMPLETE, ChainStatus.ERROR]):
                logging.error(f"Chain {chain_id} has timed out")
                self.chain_statuses[chain_id] = ChainStatus.ERROR
                return False
        return True
        
    def update_heartbeat(self, chain_id: int):
        """Update the last heartbeat time for a chain."""
        self.last_heartbeats[chain_id] = time.time()
        
    def handle_chain_failure(self, chain_id: int, error: str):
        """Handle failure of a single chain."""
        logging.error(f"Chain {chain_id} failed: {error}")
        self.chain_statuses[chain_id] = ChainStatus.ERROR
        self.terminate_processes()