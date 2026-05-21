import dataclasses
import logging
import tyro
import socket

from openpi.policies import ges_policy_2vlm_withref as _policy
from openpi.policies import policy_config_2vlm_withref as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import gesconfig_2vlm as _config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serve_policy_2vlm")


@dataclasses.dataclass
class Args:
    """2VLM deployment arguments."""

    # Training config name
    config: str = "gesvla_2vlm"

    # Checkpoint directory
    checkpoint_dir: str = ""

    # Server port
    port: int = 8000

    # Default prompt
    default_prompt: str | None = None

    # Whether to use EMA parameters
    use_ema: bool = True


def main(args: Args) -> None:
    print("=" * 60)
    print("2VLM Deployment Service")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint_dir}")

    # Load training config
    config = _config.get_config(args.config)

    # Create policy
    policy = _policy_config.create_policy_2vlm_withref(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        default_prompt=args.default_prompt,
    )
    policy_metadata = policy.metadata
    
    # Start server
    print(f"Starting server on port {args.port}...")
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
