# calvin_client.py
# Client for evaluating EVO-1 model on CALVIN benchmark via WebSocket server

import asyncio
import websockets
import numpy as np
import json
import os
import sys
import logging
import random
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import hydra
import imageio
from omegaconf import OmegaConf
from tqdm.auto import tqdm


# Add CALVIN to path
CALVIN_ROOT = Path(__file__).absolute().parents[2] / "calvin"
sys.path.insert(0, str(CALVIN_ROOT / "calvin_models"))
sys.path.insert(0, str(CALVIN_ROOT / "calvin_env"))


######################################
# Configuration
######################################
class Args:
    horizon = 12  # Number of actions to execute per inference
    max_steps = 50  # Maximum steps per episode (CALVIN uses EP_LEN=360)
    SERVER_URL = "ws://0.0.0.0:9000"
    ckpt_name = "Evo1_calvin"
    num_sequences = 1000  # Number of evaluation sequences
    SEED = 42
    dataset_path = str(CALVIN_ROOT / "dataset" / "calvin_debug_dataset")  # Update this path
    save_video = False


args = Args()

# ========= Run-specific log directory (named by timestamp) =========
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = Path(f"./log_file/{args.ckpt_name}/{RUN_TIMESTAMP}")
RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ========= Logging =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(RUN_LOG_DIR / "run.log", mode='w'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


######################################
# Evaluation Sequences (from CALVIN)
######################################
def get_sequences(num_sequences):
    """
    Get evaluation sequences from CALVIN benchmark.
    Each sequence contains an initial state and a list of 5 subtasks.
    """
    # Import from calvin_agent
    sys.path.insert(0, str(CALVIN_ROOT / "calvin_models"))
    from calvin_agent.evaluation.multistep_sequences import get_sequences as calvin_get_sequences
    return calvin_get_sequences(num_sequences)


def get_env_state_for_initial_condition(initial_condition):
    """Get robot and scene observations for a given initial condition."""
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition as calvin_get_env_state
    return calvin_get_env_state(initial_condition)


def load_task_oracle_and_annotations():
    """Load task oracle and validation annotations."""
    conf_dir = CALVIN_ROOT / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    return task_oracle, val_annotations


######################################
# Image Encoding
######################################
def encode_image_array(img_array: np.ndarray, target_size: int = 448) -> list:
    """
    Encode image array to list format for JSON transmission.
    
    Args:
        img_array: Image array (H, W, 3) in RGB format
        target_size: Target size for resizing
    
    Returns:
        Nested list of pixel values
    """
    # Resize to target size
    img = cv2.resize(img_array.astype(np.uint8), (target_size, target_size))
    # Convert RGB to BGR for consistency with Evo1 server
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img.tolist()


######################################
# Observation Processing
######################################
def obs_to_json_dict(obs: dict, prompt: str, resize_size: int = 448) -> dict:
    """
    Convert CALVIN observation to JSON-compatible dict for Evo1 server.
    
    CALVIN observation structure (from env.get_obs()):
    - rgb_obs: dict with 'rgb_static' (200x200x3), 'rgb_gripper' (84x84x3)
    - depth_obs: dict with depth images
    - robot_obs: (15,) array [tcp_pos(3), tcp_orn(3), gripper_width(1), arm_joints(7), gripper_action(1)]
    - scene_obs: (24,) array
    
    Args:
        obs: CALVIN observation dict
        prompt: Language instruction
        resize_size: Target image size
    
    Returns:
        JSON-compatible dict for Evo1 server
    """
    # Get images from rgb_obs dict
    rgb_static = obs["rgb_obs"]["rgb_static"]  # (200, 200, 3) RGB
    rgb_gripper = obs["rgb_obs"]["rgb_gripper"]  # (84, 84, 3) RGB
    
    # Create dummy third image (Evo1 expects 3 images)
    dummy_proc = np.zeros((resize_size, resize_size, 3), dtype=np.uint8)

    # print(f"observations: {obs}")
    
    # Get robot state
    # CALVIN robot_obs: [tcp_pos(3), tcp_orn(3), gripper_width(1), arm_joints(7), gripper_action(1)]
    robot_obs = obs["robot_obs"]
    
    # Convert to EVO-1 state format (7 dim): tcp_pos(3) + tcp_orn(3) + gripper_action(1)
    state = np.concatenate([
        robot_obs[:3],   # tcp_pos
        robot_obs[3:6],  # tcp_orn (euler angles)
        robot_obs[14:15],  # gripper_action
    ]).tolist()
    
    data = {
        "image": [
            encode_image_array(rgb_static, resize_size),
            encode_image_array(rgb_gripper, resize_size),
            encode_image_array(dummy_proc, resize_size),
        ],
        "state": state,
        "prompt": prompt,
        "image_mask": [1, 1, 0],  # Use static and gripper cameras
        "action_mask": [1] * 7 + [0] * 17,  # 7-dim action for CALVIN
    }
    return data


######################################
# Video Saving
######################################
def save_video_to_file(frames: list, filename: str = "simulation.mp4", fps: int = 30, save_dir: str = "videos"):
    """Save frames to video file."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    
    if len(frames) > 0:
        imageio.mimsave(filepath, frames, fps=fps)
        log.info(f"Video saved: {filepath} ({len(frames)} frames)")
    else:
        log.warning(f"No frames to save: {filepath}")


######################################
# Success Rate Calculation
######################################
def count_success(results):
    """Count success rates for each chain length."""
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results) if len(results) > 0 else 0
        step_success.append(sr)
    return step_success


def print_and_save(results, sequences, log_dir: str, epoch: str = "eval"):
    """Print and save evaluation results."""
    os.makedirs(log_dir, exist_ok=True)
    
    current_data = {}
    log.info(f"Results for {epoch}:")
    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
    log.info(f"Average successful sequence length: {avg_seq_len}")
    log.info("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        log.info(f"{i}: {sr * 100:.1f}%")

    cnt_success = Counter()
    cnt_fail = Counter()

    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    task_info = {}
    for task in total:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        log.info(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}
    current_data[epoch] = data

    # Save results
    results_file = os.path.join(log_dir, "results.json")
    previous_data = {}
    try:
        with open(results_file, "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    
    json_data = {**previous_data, **current_data}
    with open(results_file, "w") as file:
        json.dump(json_data, file, indent=2)
    
    log.info(f"Results saved to {results_file}")


######################################
# Environment Setup
######################################
def make_env(dataset_path: str):
    """Create CALVIN environment without tactile sensor (avoids numpy/networkx incompatibility)."""
    val_folder = Path(dataset_path) / "validation"
    render_conf = OmegaConf.load(val_folder / ".hydra" / "merged_config.yaml")

    # Remove tactile camera to avoid tacto -> pyrender -> networkx -> np.int error
    if "tactile" in render_conf.cameras:
        del render_conf.cameras["tactile"]
        log.info("Removed tactile sensor from camera config (numpy compatibility fix)")

    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    env = hydra.utils.instantiate(render_conf.env, show_gui=False, use_vr=False, use_scene_info=True)

    return env


######################################
# Main Evaluation Functions
######################################
async def rollout(
    ws,
    env,
    task_oracle,
    subtask: str,
    val_annotations: dict,
    horizon: int,
    max_steps: int,
    debug: bool = False
) -> tuple:
    """
    Execute a single subtask rollout.
    
    Returns:
        (success: bool, frames: list, steps_used: int)
        steps_used is the step count at success, or total steps executed on failure.
    """
    if debug:
        log.info(f"Executing subtask: {subtask}")
    
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    start_info = env.get_info()
    frames = []
    
    total_steps = 0
    for step in range(max_steps):
        # Prepare observation for server
        send_data = obs_to_json_dict(obs, lang_annotation)
        
        # Send to server and get action
        await ws.send(json.dumps(send_data))
        result = await ws.recv()
        
        try:
            action_list = json.loads(result)
            actions = np.array(action_list)
        except Exception as e:
            log.error(f"Action parsing failed: {e}")
            return False, frames, total_steps
        
        # Execute actions
        for i in range(min(horizon, len(actions))):
            raw_action = actions[i][:7].copy()  # Get first 7 dims
            
            # Extract position, orientation, and gripper action
            target_ee_pos = raw_action[:3]      # (3,) tcp position
            target_ee_orn = raw_action[3:6]     # (3,) tcp orientation (euler angles)
            
            # Convert gripper action: Evo1 outputs continuous, CALVIN expects 1/-1
            gripper_action = 1 if raw_action[6] > 0.5 else -1
            
            # CALVIN robot.apply_action expects: (target_ee_pos, target_ee_orn, gripper_action)
            action = (target_ee_pos, target_ee_orn, gripper_action)
            
            # Step environment
            obs, _, _, current_info = env.step(action)
            total_steps += 1
            
            # Collect frame for video
            if args.save_video:
                static_img = obs["rgb_obs"]["rgb_static"]
                gripper_img = cv2.resize(obs["rgb_obs"]["rgb_gripper"], (200, 200))
                frame = np.hstack([static_img, gripper_img])
                frames.append(frame)
            
            # Check if task is completed
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                if debug:
                    log.info(f"  Task {subtask} completed at step {total_steps}")
                return True, frames, total_steps
    
    if debug:
        log.info(f"  Task {subtask} failed after {total_steps} steps")
    return False, frames, total_steps


async def evaluate_sequence(
    ws,
    env,
    task_oracle,
    initial_state: dict,
    eval_sequence: list,
    val_annotations: dict,
    horizon: int,
    max_steps: int,
    seq_idx: int = 0,
    debug: bool = False
) -> tuple:
    """
    Evaluate a sequence of 5 language instructions.
    
    Returns:
        (success_count: int, all_frames: list, subtask_details: list)
        subtask_details: list of dicts with keys: subtask, success, steps, annotation
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    
    success_counter = 0
    all_frames = []
    subtask_details = []
    
    seq_str = " -> ".join(eval_sequence)
    log.info(f"[Seq {seq_idx:04d}] Sequence: {seq_str}")
    
    for task_idx, subtask in enumerate(eval_sequence):
        annotation = val_annotations[subtask][0]
        success, frames, steps_used = await rollout(
            ws, env, task_oracle, subtask, val_annotations,
            horizon, max_steps, debug
        )
        all_frames.extend(frames)
        
        detail = {
            "task_idx": task_idx + 1,
            "subtask": subtask,
            "annotation": annotation,
            "success": success,
            "steps_used": steps_used,
        }
        subtask_details.append(detail)
        
        status_str = f"✓ success (step {steps_used})" if success else f"✗ failed  (step {steps_used})"
        log.info(f"  [{task_idx + 1}/5] {subtask:<35s} | {status_str}")
        
        if success:
            success_counter += 1
        else:
            # Log remaining skipped tasks
            for remaining in eval_sequence[task_idx + 1:]:
                log.info(f"  [skip]  {remaining:<35s} | — skipped (previous task failed)")
            break
    
    log.info(f"  => Sequence result: {success_counter}/{len(eval_sequence)} subtasks completed")
    return success_counter, all_frames, subtask_details


async def run(
    SERVER_URL: str,
    dataset_path: str,
    num_sequences: int,
    horizon: int,
    max_steps: int,
    seed: int,
    save_video: bool = True,
    debug: bool = False
):
    """
    Main evaluation function.
    
    Args:
        SERVER_URL: WebSocket server URL
        dataset_path: Path to CALVIN dataset
        num_sequences: Number of evaluation sequences
        horizon: Number of actions to execute per inference
        max_steps: Maximum inference steps per subtask
        seed: Random seed
        save_video: Whether to save evaluation videos
        debug: Whether to print debug info
    """
    # ---- Header: log hyperparameters ----
    log.info("=" * 60)
    log.info("  CALVIN Evaluation Run")
    log.info("=" * 60)
    log.info(f"  Timestamp     : {RUN_TIMESTAMP}")
    log.info(f"  Checkpoint    : {args.ckpt_name}")
    log.info(f"  Server URL    : {SERVER_URL}")
    log.info(f"  Dataset path  : {dataset_path}")
    log.info(f"  Num sequences : {num_sequences}  (default: {Args.num_sequences})")
    log.info(f"  Horizon       : {horizon}  (default: {Args.horizon})")
    log.info(f"  Max steps     : {max_steps}  (default: {Args.max_steps})")
    log.info(f"  Seed          : {seed}  (default: {Args.SEED})")
    log.info(f"  Save video    : {save_video}")
    log.info(f"  Log dir       : {RUN_LOG_DIR}")
    log.info("=" * 60)
    
    # Initialize environment
    log.info("Creating CALVIN environment...")
    env = make_env(dataset_path)
    
    # Load task oracle and annotations
    log.info("Loading task oracle and annotations...")
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize_config_dir(config_dir=str(CALVIN_ROOT / "calvin_models" / "conf"))
    task_oracle, val_annotations = load_task_oracle_and_annotations()
    
    # Get evaluation sequences
    log.info("Getting evaluation sequences...")
    eval_sequences = get_sequences(num_sequences)
    
    results = []
    all_subtask_details = []  # Per-sequence subtask detail records
    video_save_dir = str(RUN_LOG_DIR / "videos")
    
    async with websockets.connect(SERVER_URL, max_size=100_000_000) as ws:
        log.info("Connected to Evo1 server")
        log.info("-" * 60)
        
        eval_iter = tqdm(eval_sequences, position=0, leave=True) if not debug else eval_sequences
        
        for seq_idx, (initial_state, eval_sequence) in enumerate(eval_iter):
            success_count, frames, subtask_details = await evaluate_sequence(
                ws, env, task_oracle, initial_state, eval_sequence,
                val_annotations, horizon, max_steps, seq_idx, debug
            )
            results.append(success_count)
            all_subtask_details.append({
                "seq_idx": seq_idx,
                "sequence": eval_sequence,
                "success_count": success_count,
                "subtasks": subtask_details,
            })
            
            # Update progress bar
            if not debug:
                eval_iter.set_description(
                    " ".join([f"{i + 1}/5: {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
                )
            
            # Save video for this sequence
            if save_video and len(frames) > 0:
                video_filename = f"seq{seq_idx:04d}_success{success_count}.mp4"
                save_video_to_file(frames, video_filename, fps=10, save_dir=video_save_dir)
    
    # Save per-sequence subtask details to JSON
    details_file = RUN_LOG_DIR / "subtask_details.json"
    with open(details_file, "w") as f:
        json.dump(all_subtask_details, f, indent=2)
    log.info(f"Per-sequence subtask details saved to {details_file}")
    
    # Print and save final results
    log.info("-" * 60)
    print_and_save(results, eval_sequences, str(RUN_LOG_DIR))
    
    env.close()
    log.info("=" * 60)
    log.info("  Evaluation Complete")
    log.info(f"  All logs saved to: {RUN_LOG_DIR}")
    log.info("=" * 60)


######################################
# Entry Point
######################################
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate EVO-1 on CALVIN benchmark")
    parser.add_argument("--server_url", type=str, default=args.SERVER_URL, help="WebSocket server URL")
    parser.add_argument("--dataset_path", type=str, default=args.dataset_path, help="Path to CALVIN dataset")
    parser.add_argument("--num_sequences", type=int, default=args.num_sequences, help="Number of evaluation sequences")
    parser.add_argument("--horizon", type=int, default=args.horizon, help="Action horizon")
    parser.add_argument("--max_steps", type=int, default=args.max_steps, help="Max inference steps per subtask")
    parser.add_argument("--save_video", action="store_true", default=args.save_video, help="Save evaluation videos")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--seed", type=int, default=args.SEED, help="Random seed")
    
    cli_args = parser.parse_args()
    
    # Set random seed   
    np.random.seed(cli_args.seed)
    random.seed(cli_args.seed)
    
    # Run evaluation
    asyncio.run(run(
        SERVER_URL=cli_args.server_url,
        dataset_path=cli_args.dataset_path,
        num_sequences=cli_args.num_sequences,
        horizon=cli_args.horizon,
        max_steps=cli_args.max_steps,
        seed=cli_args.seed,
        save_video=cli_args.save_video,
        debug=cli_args.debug
    ))
