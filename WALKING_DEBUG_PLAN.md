# Modular Walking Controller Debug Plan

## Step 1: Standing/Balance Phase (Current Step)
**Goal:** Robot stands stably, feet flat, CoM does not drift or rise/fall, no QP failures.

- **Test:** Run only the standing phase (as now).
- **Check:** CoM height and position remain nearly constant, feet do not slide, no QP errors.
- **Debug:** Tune TSID weights/gains, check contact constraints, log foot forces, etc.
- **Only move on when:** The robot can stand for several seconds with minimal drift.

---

## Step 2: Small CoM Shifts (In-Place)
**Goal:** Robot can shift its CoM slightly forward/backward/sideways without losing balance.

- **How:** In the standing phase, slowly move the CoM reference a few centimeters in X or Y.
- **Test:** Does the robot follow the CoM reference without falling or feet sliding?
- **Debug:** If it falls, tune CoM task, contact task, or posture task.
- **Only move on when:** The robot can follow small CoM shifts stably.

- stuck here rn - need to make sure robot start NOT in the air for next interation
---

## Step 3: Single Foot Lifting (Quasi-Static)
**Goal:** Robot can lift one foot off the ground and balance on the other.

- **How:** After a stable stand, slowly raise the CoM over one foot, then command the other foot to lift a few centimeters.
- **Test:** Does the robot remain stable on one foot? Does the support foot stay flat?
- **Debug:** Tune contact switching, check for QP failures, ensure foot task is not too aggressive.
- **Only move on when:** The robot can stand on one foot for a short time.

---

## Step 4: Step Placement (No Walking Yet)
**Goal:** Robot can move a foot forward and place it back down, while balancing on the other.

- **How:** After lifting a foot, move it forward (in the air), then place it back down.
- **Test:** Does the robot remain stable? Does the foot land where commanded?
- **Debug:** Tune foot trajectory, contact switching, and ensure no foot "sliding."
- **Only move on when:** The robot can step in place without falling.

---

## Step 5: Single Step (Start of Walking)
**Goal:** Robot can take a single step forward and recover balance.

- **How:** After a successful foot lift and placement, shift CoM and step with the other foot.
- **Test:** Does the robot recover balance after the step? Any QP failures?
- **Debug:** Tune DCM/ZMP controller, step timing, and foot trajectory.
- **Only move on when:** The robot can take a single step and stand stably.

---

## Step 6: Multiple Steps (Slow Walking)
**Goal:** Robot can take several steps in a row, walking slowly.

- **How:** Enable the full walking controller, but with very slow step timing and small step size.
- **Test:** Does the robot walk without falling or feet sliding?
- **Debug:** Tune all controllers, check for drift, and monitor QP status.
- **Only move on when:** The robot can walk several steps stably.

---

## Step 7: Increase Speed/Step Size
**Goal:** Gradually increase walking speed and step size to desired performance.

- **How:** Slowly increase step length, height, and decrease step time.
- **Test:** Does the robot remain stable at higher speeds?
- **Debug:** Tune as needed.

---

**Current Step:** Step 1 - Standing/Balance Phase 