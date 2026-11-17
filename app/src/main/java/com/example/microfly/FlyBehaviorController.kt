package com.example.microfly

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

enum class FlyState { IDLE, GROOM, STARTLE, EVADE, CLING }

/**
 * State machine: IDLE <-> GROOM, any -> STARTLE -> EVADE -> IDLE, wall hit -> maybe CLING -> IDLE.
 * It does not move the fly directly: it writes ACCELERATIONS into PhysicsManager
 * (steering = (desiredVelocity - velocity) * k) and exposes wing parameters for the renderer.
 */
class FlyBehaviorController(
    private val phys: PhysicsManager,
    seed: Int = 7,
) : PhysicsManager.WallHitListener {

    var state = FlyState.IDLE; private set
    /** Wing flap amplitude (degrees) and visual frequency (Hz) -> read by the renderer. */
    var wingAmp = 4f; private set
    var wingHz = 6f; private set

    private val rnd = Random(seed)
    private var t = 0f
    private var dur = 2f
    private var jinkT = 0f
    private var twitchT = 1f
    private var twitchLeft = 0f
    // current heading (unit vector) and wall normal of last cling
    private var hx = 1f; private var hy = 0f; private var hz = 0f
    private var wnx = 0f; private var wny = 0f; private var wnz = 0f

    init { phys.listener = this }

    /** Called by touch: (dx,dy,dz) is the direction pointing AWAY from the touch ray. */
    fun startle(dx: Float, dy: Float, dz: Float) {
        phys.clingTimeLeft = 0f
        setHeading(dx + jitter(0.25f), dy + 0.25f + jitter(0.25f), dz + jitter(0.25f))
        phys.setVelocity(hx * STARTLE_SPEED, hy * STARTLE_SPEED, hz * STARTLE_SPEED)
        enter(FlyState.STARTLE, 0.30f)
    }

    /** Called continuously while the user drags ON the fly: keep it panicking. */
    fun panic(dx: Float, dy: Float, dz: Float) {
        if (state == FlyState.EVADE || state == FlyState.STARTLE) { dur = t + 1.2f; return }
        startle(dx, dy, dz)
    }

    fun update(dt: Float) {
        t += dt
        var targetAmp: Float
        var targetHz: Float

        when (state) {
            FlyState.IDLE -> {
                steer(hx * 0.6f, hy * 0.2f, hz * 0.6f, 1.5f)
                // random short wing twitches
                twitchT -= dt
                if (twitchT <= 0f) { twitchLeft = 0.12f; twitchT = 1.5f + rnd.nextFloat() * 3f }
                twitchLeft -= dt
                targetAmp = if (twitchLeft > 0f) 28f else 3f
                targetHz = if (twitchLeft > 0f) 14f else 4f
                if (t > dur) {
                    if (rnd.nextFloat() < 0.3f) enter(FlyState.GROOM, 1f + rnd.nextFloat() * 1.5f)
                    else { randomHeading(); enter(FlyState.IDLE, 1.5f + rnd.nextFloat() * 2f) }
                }
            }
            FlyState.GROOM -> {
                steer(0f, 0f, 0f, 5f)                    // stop and "clean"
                targetAmp = 2f + 2f * abs(sin(t * 18f))  // tiny jitter
                targetHz = 4f
                if (t > dur) enter(FlyState.IDLE, 1f)
            }
            FlyState.STARTLE -> {                         // ballistic burst, no steering
                phys.ax = 0f; phys.ay = 0f; phys.az = 0f
                phys.drag = 0.4f
                targetAmp = 55f; targetHz = 24f
                if (t > dur) enter(FlyState.EVADE, 1.5f + rnd.nextFloat())
            }
            FlyState.EVADE -> {
                jinkT -= dt
                if (jinkT <= 0f) { jink(); jinkT = 0.15f + rnd.nextFloat() * 0.2f }
                steer(hx * EVADE_SPEED, hy * EVADE_SPEED, hz * EVADE_SPEED, 4f)
                targetAmp = 50f; targetHz = 22f
                if (t > dur) enter(FlyState.IDLE, 1f)
            }
            FlyState.CLING -> {
                phys.ax = 0f; phys.ay = 0f; phys.az = 0f
                twitchT -= dt
                targetAmp = if (twitchT < 0.1f) 15f else 1f
                targetHz = 10f
                if (twitchT <= 0f) twitchT = 0.5f + rnd.nextFloat()
                if (!phys.isClinging) {                   // release: push off the glass
                    setHeading(wnx + jitter(0.4f), wny + jitter(0.4f), wnz + jitter(0.4f))
                    phys.setVelocity(hx * 2.5f, hy * 2.5f, hz * 2.5f)
                    enter(FlyState.IDLE, 1.5f)
                }
            }
        }
        // smooth wing parameter changes (no abrupt jumps)
        val s = minOf(1f, dt * 12f)
        wingAmp += (targetAmp - wingAmp) * s
        wingHz += (targetHz - wingHz) * s
    }

    override fun onWallHit(nx: Float, ny: Float, nz: Float, impactSpeed: Float) {
        wnx = nx; wny = ny; wnz = nz
        // Bounce is already done by physics. Sometimes the fly then sticks to the glass.
        if (state != FlyState.CLING && impactSpeed > 1f && rnd.nextFloat() < 0.2f) {
            phys.clingTimeLeft = 0.5f + rnd.nextFloat() * 0.9f
            twitchT = 0.3f
            enter(FlyState.CLING, 99f)
        } else if (state == FlyState.IDLE || state == FlyState.GROOM) {
            // reflect wander heading so it turns away from the wall
            setHeading(nx + jitter(0.5f), ny * 0.3f, nz + jitter(0.5f))
        }
    }

    // ---- helpers ---------------------------------------------------------------------------
    private fun enter(s: FlyState, duration: Float) {
        state = s; t = 0f; dur = duration
        if (s == FlyState.IDLE || s == FlyState.GROOM) phys.drag = 1.5f
        if (s == FlyState.EVADE) { phys.drag = 0.8f; jinkT = 0f }
    }

    /** Acceleration toward a desired velocity (simple proportional steering). */
    private fun steer(dvx: Float, dvy: Float, dvz: Float, k: Float) {
        phys.ax = (dvx - phys.vx) * k
        phys.ay = (dvy - phys.vy) * k
        phys.az = (dvz - phys.vz) * k
    }

    /** Sudden turn (yaw +-40..80 deg) plus an altitude change. */
    private fun jink() {
        val a = (40f + rnd.nextFloat() * 40f) * DEG * (if (rnd.nextBoolean()) 1f else -1f)
        val c = cos(a); val s = sin(a)
        val nx = hx * c + hz * s
        val nz = -hx * s + hz * c
        setHeading(nx, hy + (rnd.nextFloat() - 0.5f) * 1.2f, nz)
    }

    private fun randomHeading() =
        setHeading(rnd.nextFloat() - 0.5f, (rnd.nextFloat() - 0.5f) * 0.4f, rnd.nextFloat() - 0.5f)

    private fun setHeading(x: Float, y: Float, z: Float) {
        val l = sqrt(x * x + y * y + z * z)
        if (l < 1e-4f) { hx = 1f; hy = 0f; hz = 0f; return }
        hx = x / l; hy = y / l; hz = z / l
    }

    private fun jitter(a: Float) = (rnd.nextFloat() - 0.5f) * 2f * a

    companion object {
        const val STARTLE_SPEED = 9f
        const val EVADE_SPEED = 5f
        private const val DEG = (Math.PI / 180.0).toFloat()
    }
}
