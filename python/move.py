import pi_servo_hat
import time
import sys
import curses

def runExample():
	mySensor = pi_servo_hat.PiServoHat()

	mySensor.restart()
  
	# Test Run
	#########################################
	# Moves servo position to 0 degrees (1ms), Channel 0
	mySensor.move_servo_position(14, 90)

	# Moves servo position to 90 degrees (2ms), Channel 0
	mySensor.move_servo_position(15, 90)

	time.sleep(5)
	mySensor.move_servo_position(14,0)
	mySensor.move_servo_position(15,0)

	time.sleep(5)

def moveEyes(mySensor, position, skew):
	mySensor.move_servo_position(14,position+skew)
	mySensor.move_servo_position(15,position-skew)

def moveWithKeys():
	position = 45
	step = 15
	skew = 0
	mySensor = pi_servo_hat.PiServoHat()

	mySensor.restart()

	# get the curses screen window
	screen = curses.initscr()

	# turn off input echoing
	curses.noecho()

	# respond to keys immediately (don't wait for enter)
	curses.cbreak()

	# map arrow keys to special values
	screen.keypad(True)

	try:
		while True:
			char = screen.getch()
			if char == ord('q'):
				break
			elif char == curses.KEY_LEFT:
				# print doesn't work with curses, use addstr instead
				if position + step + abs(skew) <= 145:
					position = position + step
				else:
					position = 145
				screen.addstr(0, 0,str(position))
				moveEyes(mySensor,position,skew)
			elif char == curses.KEY_RIGHT:
				if position - step - abs(skew)  >= -45 :
					position = position-step
				else:
					position = -45
				screen.addstr(0, 0, str(position))
				moveEyes(mySensor,position,skew)
			elif char == curses.KEY_UP:
				skew = skew + 5
				moveEyes(mySensor,position,skew)
				screen.addstr(0, 0, str(skew))
			elif char == curses.KEY_DOWN:
				skew = skew - 5
				moveEyes(mySensor,position,skew)
				screen.addstr(0, 0, str(skew))
	finally:
    		# shut down cleanly
    		curses.nocbreak()
    		screen.keypad(0)
    		curses.echo()
    		curses.endwin()

def main():
	print("Hello World!")
	moveWithKeys()
	#while 1<2:
	#	runExample()

	

if __name__ == "__main__":
    main()
