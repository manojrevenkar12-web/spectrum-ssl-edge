// Tests check what renders, not how it animates; happy-dom's Web Animations cancel asynchronously
// after a test's cleanup, which would surface as unhandled rejections.
import { MotionGlobalConfig } from 'motion/react';

MotionGlobalConfig.skipAnimations = true;
