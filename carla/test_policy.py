import cv2
import os

from carla.evaluate_agent import load_policy_and_env, eval_policy


def write_video(images, fn_output='video.mp4', frame_rate=20, overwrite=False):
    """Takes a list of images and interprets them as frames for a video.

    Source: http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html
    """
    height, width, _ = images[0].shape

    if overwrite:
        if os.path.exists(fn_output):
            os.remove(fn_output)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(fn_output, fourcc, frame_rate, (width, height))

    for cur_image in images:
        frame = cv2.resize(cur_image, (width, height))
        out.write(frame)  # Write out frame to video

    # Release everything if job is finished
    out.release()

    return fn_output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i',  default="last")
    parser.add_argument('--context_config',  default=None)
    parser.add_argument('--store_video', action='store_true')
    parser.add_argument('--print_model', action='store_true')
    args = parser.parse_args()
    env, get_action, model, _ = load_policy_and_env(args.fpath, args.context_config, itr=args.itr)

    if args.print_model:
        print(model)
    else:
        n_contexts = env.envs[0].unwrapped.n_contexts
        _, _, _, _, observations = eval_policy(env, get_action, args.episodes, n_contexts=n_contexts,
                                               render=not (args.norender), store_video=args.store_video)
        if args.store_video:
            write_video(observations, fn_output=os.path.join(args.fpath, "video.mp4"), frame_rate=5, overwrite=True)
