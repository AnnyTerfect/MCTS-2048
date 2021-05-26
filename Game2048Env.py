import copy
import math
import random
import sys
import time
import warnings
import platform, multiprocessing

import gym  
import numpy as np
from PyQt5.QtWidgets import (QMainWindow,QWidget,QPushButton,QDialog,
    QApplication,QDesktopWidget,QLabel,QMessageBox)
from PyQt5.QtGui import QIcon,QFont
from PyQt5.QtCore import Qt


class Game2048Env(gym.Env):
    # render: whether to display gui
    def __init__(self, render:bool=True):
        super(Game2048Env, self).__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=2048, shape=(4,4))
        self.render = render
        self.done = False
        self.state = [[0, 0, 0, 0] for i in range(4)]
        self.score_dict = {}
        for num in range(1, 12):
            self.score_dict[2 ** num] = 2 ** (num - 1)
        self.episode_length = 0
        self.info = {}
        self.port = None
        if self.render:
            self._run()
            msg = self.port.recv()
            assert msg == 'init_ok'

    def reset(self):
        self.state = [[0, 0, 0, 0] for i in range(4)]
        self._generateNew(2)
        self.done = False
        self.info = {}
        self.score = 0
        self.episode_length = 0

    # returns obs, rew, done, info
    def step(self, action: int):
        assert self.action_space.contains(action), 'Illegal Action!'
        prev_score = self.score

        def _moveLeft():
            self.prev_board=copy.deepcopy(self.state)
            self.episode_length += 1
            for i in range(4):
                tmp = []
                for j in range(4):
                    if self.state[i][j] != 0:
                        tmp.append(self.state[i][j])
                for k in range(len(tmp) - 1):
                    if tmp[k] != 0 and tmp[k] == tmp[k + 1]:
                        tmp[k] *= 2
                        self.score += self.score_dict[tmp[k]]
                        tmp = tmp[:k + 1] + tmp[k + 2:] + [0]
                tmp += [0] * (4 - len(tmp))
                self.state[i] = tmp
            self._checkBoard()
        def _moveRight():
            self.prev_board=copy.deepcopy(self.state)
            self.episode_length += 1
            for i in range(4):
                tmp = []
                for j in range(4):
                    if self.state[i][j] != 0:
                        tmp.append(self.state[i][j])
                for k in range(len(tmp) - 1, 0, -1):
                    if tmp[k] != 0 and tmp[k] == tmp[k - 1]:
                        tmp[k] *= 2
                        self.score += self.score_dict[tmp[k]]
                        tmp = [0] + tmp[:k - 1] + tmp[k:]
                tmp = [0] * (4 - len(tmp)) + tmp
                self.state[i] = tmp
            self._checkBoard()
        def _moveUp():
            self.prev_board = copy.deepcopy(self.state)
            self.episode_length += 1
            for j in range(4):
                tmp = []
                for i in range(4):
                    if self.state[i][j] != 0:
                        tmp.append(self.state[i][j])
                for k in range(len(tmp) - 1):
                    if tmp[k] != 0 and tmp[k] == tmp[k + 1]:
                        tmp[k] *= 2
                        self.score += self.score_dict[tmp[k]]
                        tmp = tmp[:k + 1] + tmp[k + 2:] + [0]
                tmp += [0] * (4 - len(tmp))
                for i in range(4):
                    self.state[i][j] = tmp[i]
        def _moveDown():
            self.prev_board = copy.deepcopy(self.state)
            self.episode_length += 1
            for j in range(4):
                tmp = []
                for i in range(4):
                    if self.state[i][j] != 0:
                        tmp.append(self.state[i][j])
                for k in range(len(tmp) - 1, 0, -1):
                    if tmp[k] != 0 and tmp[k] == tmp[k - 1]:
                        tmp[k] *= 2
                        self.score += self.score_dict[tmp[k]]
                        tmp = [0] + tmp[:k - 1] + tmp[k:]
                tmp = [0] * (4 - len(tmp)) + tmp
                for i in range(4):
                    self.state[i][j] = tmp[i]
        
        if action == 0:
            _moveUp()
        elif action == 1:
            _moveDown()
        elif action == 2:
            _moveLeft()
        elif action == 3:
            _moveRight()
        self._checkBoard()
        if self.render:
            self.port.send(("step",(self.score, self.state)))
            msg = self.port.recv()
            assert msg == 'step_ok'
        return self.state, self.score - prev_score, self.done, self.info
   
    def close(self):
        if self.port is not None:
            self.port.send(("close", None))
            self.port.close()
            self.gui_process.terminate()
            self.gui_process.join()

    def seed(self, seed):
        random.seed(seed)

    # you can modify the following two functions if you really need them, but they are neither tested nor recommended
    # UNTESTED
    def setRender(self, render:bool):
        warnings.warn("Not recommended API", UserWarning)
        if self.render and not render:
            self.close()
            self.port = None
            self.gui_process = None
        if not self.render and render:
            self._run()
            msg = self.port.recv()
            assert msg == 'init_ok'
        self.render = render

    # UNTESTED
    def set(self, state, score=0):
        warnings.warn("Not recommended API", UserWarning)
        self.state = copy.deepcopy(state)
        self.score = score
        if self.render:
            self.port.send(("set",(self.state,self.score)))
            msg = self.port.recv()
            assert msg == 'set_ok'
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != 'gui_process' and k != 'port':
                setattr(result, k, copy.deepcopy(v, memo))
        result.render = False
        return result

    def _run(self):
        from multiprocessing import Process, Pipe
        port, remote_port = Pipe(duplex=True)
        self.port = port
        self.gui_process = Process(target=self._initGUI, args=(remote_port, port, ))
        self.gui_process.start()
        remote_port.close()

    def _initGUI(self, port, remote_port):
        app=QApplication(sys.argv)
        qt_game=Game2048GUI(port, remote_port)
        sys.exit(app.exec_())



    def _checkBoard(self):
        for i in range(4):
            for j in range(4):
                if self.state[i][j] == 2048:
                    self.done = True
                    self.info['episode_length'] = self.episode_length
                    self.info['success'] = True
                    self.info['total_score'] = self.score
        if self.prev_board != self.state and self._getAvailablePos(self.state):
            self._generateNew(1)
        elif not self._getAvailablePos(self.state):  
            def _canMove(board):
                for i in range(1, len(board) - 1):
                    for j in range(1, len(board[0]) - 1):
                        if board[i][j] == board[i - 1][j] or board[i][j] == board[i + 1][j] or board[i][j] == board[i][j - 1] or \
                                board[i][j] == board[i][j + 1]:
                            return True
                if board[0][0] == board[0][1] or board[0][0] == board[1][0]:
                    return True
                if board[3][0] == board[3][1] or board[3][0] == board[2][0]:
                    return True
                if board[0][3] == board[1][3] or board[0][3] == board[0][2]:
                    return True
                if board[3][3] == board[3][2] or board[3][3] == board[2][3]:
                    return True
                if board[1][0] == board[2][0] or board[0][1] == board[0][2] or board[3][1] == board[3][2] or board[3][1] == \
                        board[3][2]:
                    return True
                return False
            if not _canMove(self.state):
                self.done = True
                self.info['episode_length'] = self.episode_length
                self.info['total_score'] = self.score
                if self.info.get('success') is None:
                    self.info['success'] = False
                

    def _generateNew(self, num):
        sample = random.sample(self._getAvailablePos(self.state), num)
        for i in range(num):
            x, y = sample[i]
            self.state[x][y] = 2

    def _getAvailablePos(self, arr):
        res = []
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                if arr[i][j] == 0:
                    res.append([i, j])
        return res


class Game2048GUI(QMainWindow):
    def __init__(self, port, remote_port):
        super(Game2048GUI, self).__init__()
        self.port = port
        remote_port.close()
        self._initUI()
        if self.port is not None:
            self._mainLoop()

    def _initUI(self):
        self.color_dict={}
        self.score = 0
        self.state = [[0, 0, 0, 0] for i in range(4)]

        # you can change to whatever you like :)
        self.color_dict[0]='white'
        col = ['lightgrey','grey', 'pink', 'yellow', 'brown', 'red', 'violet', 'crimson', 'fuchsia'] * 100
        for num in range(1, 11):
            self.color_dict[2 ** num] = col[num - 1]

        self.state_font = QFont('Consolas', 20)
        self.scoreboard_font = QFont('Consolas', 16)
        self.button_font = QFont('Consolas', 12)

        self.setGeometry(300, 300, 500, 550)
        self.setWindowTitle("2048")
        self.center()

        self.scoreboard=QLabel(self)
        self.scoreboard.resize(150, 80)
        self.scoreboard.move(30, 0)
        self.scoreboard.setFont(self.scoreboard_font)
        self.scoreboard.setText('Score : {}'.format(self.score))

        # restart_button = QPushButton('Restart', self)
        # restart_button.setFont(self.button_font)
        # restart_button.resize(80, 30)
        # restart_button.move(380, 25)
        # restart_button.clicked.connect(self._restart)
        # restart_button.setFocusPolicy(Qt.NoFocus)

        self._initBoardUI()
        self._updateBoardUI()
        self.show()

    def _mainLoop(self):
        self.port.send('init_ok')
        while True:
            QApplication.processEvents()
            cmd, data = self.port.recv()
            if cmd == "step":
                self.score, self.state = data
                self._updateBoardUI()
                self.port.send("step_ok")
            if cmd == "set":
                self.score, self.state = data
                self._updateBoardUI()
                self.port.send("set_ok")
            if cmd == "close":
                self.port.close()
                self.exit(0)

    def _initBoardUI(self):
        self.scoreboard.setText('Score : {}'.format(0))
        self.state_labels = [[0, 0, 0, 0] for i in range(4)]
        for i in range(4):
            for j in range(4):
                self.state_labels[i][j] = QLabel(self)
                self.state_labels[i][j].setFont(self.state_font)
                self.state_labels[i][j].setAlignment(Qt.AlignCenter)
                self.state_labels[i][j].resize(100, 100)
                self.state_labels[i][j].move(110 * j + 30, 110 *i+70)

    def _updateBoardUI(self):
        for i in range(4):
            for j in range(4):
                col=self.color_dict[self.state[i][j]]
                if self.state[i][j] == 0:
                    txt=''
                else:
                    txt=str(self.state[i][j])
                self.state_labels[i][j].setText(txt)
                self.state_labels[i][j].setStyleSheet('background-color:%s'%col)
        self.scoreboard.setText('Score : {}'.format(self.score))

    def center(self):
        qr=self.frameGeometry()
        cp=QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self,event):
        QMessageBox.information(self, "Hint", "Please close the window from the console")
        event.ignore()

if __name__ == '__main__':
    #env = Game2048Env(True) 
    env = Game2048Env(False) 
    # you can fix the seed for debugging, but your agent SHOULD NOT overfit to the env of a certain seed
    env.seed(0)
    # render is automatically set to False for copied envs
    # remember to call reset() before calling step()
    obs = env.reset()
    done = False

    while not done:
        action = random.randint(0, 3)
        obs_new, rew, done, info = env.step(action)
        print(rew, done, info)
    print('game done')
    # remember to close the env, but you can always let resources leak on your own computer :|
    env.close()
