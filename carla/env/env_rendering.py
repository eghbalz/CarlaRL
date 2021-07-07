import math
import numpy as np

from contextual_gridworld.environment.colors import IDX_TO_COLOR, COLOR_TO_IDX
from contextual_gridworld.environment.env import load_context_config, Grid, WorldObj


class EnvRenderer:

    def __init__(self, total_objects, grid_size=8, tile_size=8,  context_config="color_contexts.yaml"):

        self.tile_size = tile_size
        self.total_objects = total_objects

        self.contexts, self.subdivs = load_context_config(context_config)
        self.context = self.contexts[0]

        self.obstacles = []
        self.goodies = []

        # Environment configuration
        self.width = grid_size
        self.height = grid_size

        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        self.grid = None
        self.empty_grid()

    def empty_grid(self):
        # Create an empty grid
        self.grid = Grid(self.width, self.height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, self.width, self.height)

    def get_empty_positions(self, agent_pos, agent_dir):

        self.empty_grid()
        self.agent_pos = np.round(agent_pos).astype(np.int32)
        self.agent_dir = agent_dir

        empty_positions = []
        empty_positions_transformed = []
        grid = self.grid

        for j in range(0, grid.height):
            for i in range(0, grid.width):
                cell = grid.get(i, j)

                agent_here = agent_pos[0] == i and agent_pos[1] == j

                if not agent_here and cell is None:

                    pos = np.asarray([i, j])

                    empty_positions.append(pos)

                    theta = np.deg2rad(-self.agent_dir * 90)
                    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

                    pos = np.dot(rotation, pos) % (self.width - 1)

                    empty_positions_transformed.append(pos)

        self.agent_pos = None
        self.agent_dir = None
        return empty_positions, empty_positions_transformed

    def transfer_positions(self, pos):

        # undo rotation
        theta = np.deg2rad(self.agent_dir*90)
        rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        pos = np.dot(rotation, pos) % (self.width - 1)

        return np.round(pos).astype(dtype=np.int32)

    def render_gt(self, gt, agent_pos, agent_dir):

        self.empty_grid()

        self.agent_pos = np.round(agent_pos).astype(np.int32)
        self.agent_dir = agent_dir

        agent_gt = gt[::self.total_objects]
        goal_gt = gt[1::self.total_objects]

        n_goodies = n_obstacles = (self.total_objects - 2) // 2

        goodies_gt = []
        for i in range(2, 2+n_goodies):
            goodies_gt.append(gt[i::self.total_objects])

        obstacles_gt = []
        for i in range(2+n_goodies, 2+n_goodies+n_obstacles):
            obstacles_gt.append(gt[i::self.total_objects])

        goal_x, goal_y, color_idx = goal_gt

        if goal_x >= 0:
            goal_x, goal_y = self.transfer_positions(np.asarray([goal_x, goal_y]))
            self.grid.set(goal_x, goal_y, WorldObj('goal', IDX_TO_COLOR[int(color_idx)]))

        for goodie in goodies_gt:

            pos_x, pos_y, color_idx = goodie
            if pos_x >= 0:
                pos_x, pos_y = self.transfer_positions(np.asarray([pos_x, pos_y]))
                self.grid.set(pos_x, pos_y, WorldObj('goodie', IDX_TO_COLOR[int(color_idx)]))

        for obstacle in obstacles_gt:

            pos_x, pos_y, color_idx = obstacle
            if pos_x >= 0:
                pos_x, pos_y = self.transfer_positions(np.asarray([pos_x, pos_y]))
                self.grid.set(pos_x, pos_y, WorldObj('obstacle', IDX_TO_COLOR[int(color_idx)]))


        # Render the whole grid
        img = self.grid.render(
            self.tile_size,
            self.agent_pos,
            self.agent_dir,
            # 0,
            subdivs=self.subdivs,
            agent_id=IDX_TO_COLOR[int(agent_gt[2])]
        )

        img = np.rot90(img, k=self.agent_dir)

        return img


def get_gt_factors(env, total_objects, max_n_goodies, max_n_obstacles):
    """
     this is the order of variables:

     x,y,color

     and this is the order of objects:

     agent, goal, goodie1, goodie2, obstacle1, obstacle2

     in details, this is how the array is filled:

     agent_x, goal_x, ...., agent_y, goal_y,...., agent_colour, goal_colour ....

     :return:
     """

    gt = np.ones(3 * total_objects) * -1

    # rotate positions according to agent direction
    theta = np.deg2rad(-env.agent_dir*90)
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

    agent_pos = np.dot(rotation, env.agent_pos) % (env.width - 1)

    goal_pos = np.dot(rotation, env.goal_pos) % (env.width - 1)

    goodies_pos = []
    for goodie in env.goodies:
        goodies_pos.append(np.dot(rotation, goodie.cur_pos) % (env.width - 1))

    obstacles_pos = []
    for obstacle in env.obstacles:
        obstacles_pos.append(np.dot(rotation, obstacle.cur_pos) % (env.width - 1))

    offset = 0

    # place agent
    gt[offset] = agent_pos[0]
    gt[total_objects + offset] = agent_pos[1]
    gt[2*total_objects + offset] = COLOR_TO_IDX[env.context['agent']]
    offset += 1

    # place goal
    if goal_pos is not None:
        gt[offset] = goal_pos[0]
        gt[total_objects + offset] = goal_pos[1]
        gt[2*total_objects + offset] = COLOR_TO_IDX[env.context['goal']]
    offset += 1

    for idx in range(max_n_goodies):

        if len(goodies_pos) > idx:
            gt[offset] = goodies_pos[idx][0]
            gt[total_objects + offset] = goodies_pos[idx][1]
            gt[2*total_objects + offset] = COLOR_TO_IDX[env.context['goodie']]
        offset += 1

    for idx in range(max_n_obstacles):

        if len(obstacles_pos) > idx:
            gt[offset] = obstacles_pos[idx][0]
            gt[total_objects + offset] = obstacles_pos[idx][1]
            gt[2*total_objects + offset] = COLOR_TO_IDX[env.context['obstacle']]
        offset += 1

    return gt, env.agent_pos, env.agent_dir
