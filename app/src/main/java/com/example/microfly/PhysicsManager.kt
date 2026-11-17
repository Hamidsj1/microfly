package com.example.microfly

/**
 * Minimal rigid-body physics for ONE sphere (the fly) inside an axis-aligned box (the "glass walls").
 * Everything is primitive floats -> zero allocations per frame.
 *
 * Integration: semi-implicit (symplectic) Euler with a FIXED timestep (accumulator pattern),
 * so the simulation is deterministic and stable even when the frame time jitters.
 *      v += a * dt
 *      v *= 1 / (1 + drag * dt)      (implicit linear air drag, unconditionally stable)
 *      p += v * dt
 */
class PhysicsManager(
    private val halfX: Float,
    private val halfY: Float,
    private val halfZ: Float,
    val radius: Float,
) {
    fun interface WallHitListener {
        /** Normal points INTO the box (away from the wall). */
        fun onWallHit(nx: Float, ny: Float, nz: Float, impactSpeed: Float)
    }

    var px = 0f; var py = 0f; var pz = 0f
    var vx = 0f; var vy = 0f; var vz = 0f
    /** Acceleration set by the behaviour controller each frame. */
    var ax = 0f; var ay = 0f; var az = 0f

    var drag = 1.0f
    var restitution = 0.6f          // energy kept after bounce (0.6 -> ~40% speed loss)
    var clingTimeLeft = 0f
    val isClinging get() = clingTimeLeft > 0f
    var listener: WallHitListener? = null

    private var accumulator = 0f

    fun step(frameDt: Float) {
        accumulator += frameDt
        // Cap substeps so a long hitch can't cause a spiral of death.
        var n = 0
        while (accumulator >= FIXED_DT && n < 6) {
            integrate(FIXED_DT)
            accumulator -= FIXED_DT
            n++
        }
        if (n == 6) accumulator = 0f
    }

    private fun integrate(dt: Float) {
        if (isClinging) {               // stuck to the glass: no motion
            clingTimeLeft -= dt
            vx = 0f; vy = 0f; vz = 0f
            return
        }
        vx += ax * dt; vy += ay * dt; vz += az * dt
        val k = 1f / (1f + drag * dt)
        vx *= k; vy *= k; vz *= k
        px += vx * dt; py += vy * dt; pz += vz * dt
        collide()
    }

    /**
     * Sphere vs. AABB-interior collision. For every axis: if the sphere crosses the wall,
     *  1) push it back to the wall (no tunnelling / no passing through),
     *  2) reflect the velocity component: v = -v * restitution.
     */
    private fun collide() {
        val lx = halfX - radius; val ly = halfY - radius; val lz = halfZ - radius
        if (px > lx)  { px = lx;  hit(-1f, 0f, 0f, vx);  vx = -vx * restitution }
        if (px < -lx) { px = -lx; hit(1f, 0f, 0f, -vx);  vx = -vx * restitution }
        if (py > ly)  { py = ly;  hit(0f, -1f, 0f, vy);  vy = -vy * restitution }
        if (py < -ly) { py = -ly; hit(0f, 1f, 0f, -vy);  vy = -vy * restitution }
        if (pz > lz)  { pz = lz;  hit(0f, 0f, -1f, vz);  vz = -vz * restitution }
        if (pz < -lz) { pz = -lz; hit(0f, 0f, 1f, -vz);  vz = -vz * restitution }
    }

    private fun hit(nx: Float, ny: Float, nz: Float, speed: Float) {
        if (speed > 0.05f) listener?.onWallHit(nx, ny, nz, speed)
    }

    fun setVelocity(x: Float, y: Float, z: Float) { vx = x; vy = y; vz = z }

    companion object { const val FIXED_DT = 1f / 120f }
}
