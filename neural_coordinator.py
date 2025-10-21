"""
Neural Network-like Coordinator System
Components communicate and coordinate actions through event-driven architecture
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Callable, Any
from collections import defaultdict
import uuid

class NeuralCoordinator:
    """
    Central nervous system for the application
    Enables components to communicate, share state, and coordinate actions
    """
    
    def __init__(self):
        self.neurons = {}  # Component registry
        self.synapses = defaultdict(list)  # Event listeners (connections between neurons)
        self.state = {}  # Shared state across all components
        self.health_status = {}  # Health status of each neuron
        self.event_queue = asyncio.Queue()
        self.running = False
        self.task = None
        
    def register_neuron(self, name: str, component: Any, health_check: Callable = None):
        """Register a component (neuron) in the network"""
        self.neurons[name] = {
            "component": component,
            "health_check": health_check,
            "registered_at": datetime.utcnow(),
            "last_active": datetime.utcnow()
        }
        self.health_status[name] = "healthy"
        logging.info(f"🧠 Neuron registered: {name}")
        
    def connect_synapse(self, event_type: str, handler: Callable):
        """Create a synapse (event listener) between neurons"""
        self.synapses[event_type].append(handler)
        logging.info(f"🔗 Synapse connected: {event_type}")
        
    async def fire_signal(self, event_type: str, data: Dict = None):
        """Fire a signal through the neural network"""
        event = {
            "id": str(uuid.uuid4()),
            "type": event_type,
            "data": data or {},
            "timestamp": datetime.utcnow(),
            "processed": False
        }
        await self.event_queue.put(event)
        logging.debug(f"⚡ Signal fired: {event_type}")
        
    async def process_signals(self):
        """Process signals through the neural network"""
        while self.running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Execute all handlers for this event type
                handlers = self.synapses.get(event["type"], [])
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event["data"])
                        else:
                            handler(event["data"])
                    except Exception as e:
                        logging.error(f"❌ Synapse error in {event['type']}: {str(e)}")
                
                event["processed"] = True
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"❌ Neural network error: {str(e)}")
                
    async def health_check_loop(self):
        """Continuously monitor health of all neurons"""
        while self.running:
            try:
                for name, neuron in self.neurons.items():
                    if neuron["health_check"]:
                        try:
                            if asyncio.iscoroutinefunction(neuron["health_check"]):
                                is_healthy = await neuron["health_check"]()
                            else:
                                is_healthy = neuron["health_check"]()
                            
                            old_status = self.health_status[name]
                            new_status = "healthy" if is_healthy else "unhealthy"
                            
                            if old_status != new_status:
                                self.health_status[name] = new_status
                                await self.fire_signal("health_change", {
                                    "neuron": name,
                                    "old_status": old_status,
                                    "new_status": new_status
                                })
                                logging.warning(f"⚠️ Health change: {name} -> {new_status}")
                            
                        except Exception as e:
                            logging.error(f"❌ Health check failed for {name}: {str(e)}")
                            self.health_status[name] = "error"
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logging.error(f"❌ Health check loop error: {str(e)}")
                
    def update_shared_state(self, key: str, value: Any):
        """Update shared state that all neurons can access"""
        old_value = self.state.get(key)
        self.state[key] = value
        
        # Fire signal about state change
        asyncio.create_task(self.fire_signal("state_change", {
            "key": key,
            "old_value": old_value,
            "new_value": value
        }))
        
    def get_shared_state(self, key: str, default: Any = None):
        """Get shared state"""
        return self.state.get(key, default)
        
    async def start(self):
        """Start the neural network"""
        self.running = True
        self.task = asyncio.create_task(self.process_signals())
        self.health_task = asyncio.create_task(self.health_check_loop())
        logging.info("🧠 Neural network started")
        
    async def stop(self):
        """Stop the neural network"""
        self.running = False
        if self.task:
            self.task.cancel()
        if self.health_task:
            self.health_task.cancel()
        logging.info("🧠 Neural network stopped")
        
    def get_network_status(self) -> Dict:
        """Get status of entire neural network"""
        return {
            "neurons": list(self.neurons.keys()),
            "health_status": self.health_status,
            "synapses": {k: len(v) for k, v in self.synapses.items()},
            "shared_state": self.state,
            "running": self.running
        }

# Global coordinator instance
coordinator = NeuralCoordinator()
