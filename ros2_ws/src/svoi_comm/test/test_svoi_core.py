import unittest

from svoi_comm.svoi_core import EventTriggeredGate, PeriodicPhaseSchedule


class PeriodicPhaseScheduleTest(unittest.TestCase):
    def test_four_balanced_phases(self):
        schedules = [PeriodicPhaseSchedule(4.0, float(i), 100.0) for i in range(4)]
        fire_times = [[] for _ in schedules]
        for step in range(81):
            now = 100.0 + step * 0.05
            for robot_id, schedule in enumerate(schedules):
                if schedule.due_cycle(now) is not None:
                    fire_times[robot_id].append(round(now - 100.0, 2))
        self.assertEqual(fire_times[0], [0.0, 4.0])
        self.assertEqual(fire_times[1], [1.0])
        self.assertEqual(fire_times[2], [2.0])
        self.assertEqual(fire_times[3], [3.0])

    def test_late_callback_does_not_burst(self):
        schedule = PeriodicPhaseSchedule(4.0, 1.0, 100.0)
        self.assertEqual(schedule.due_cycle(109.2), 2)
        self.assertIsNone(schedule.due_cycle(109.3))
        self.assertEqual(schedule.due_cycle(113.0), 3)

    def test_periodic_tick_phase_offset(self):
        gate = EventTriggeredGate(mode="periodic", interval=20, phase_offset=5)
        fired = [step for step in range(25) if gate.should_fire(step, 0.0)]
        self.assertEqual(fired, [5])

    def test_invalid_period(self):
        with self.assertRaises(ValueError):
            PeriodicPhaseSchedule(0.0)


if __name__ == "__main__":
    unittest.main()
