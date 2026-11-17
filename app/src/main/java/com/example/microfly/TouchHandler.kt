package com.example.microfly

import android.opengl.Matrix
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.PointerEvent
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.pointerInput
import com.google.android.filament.Camera
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * Gestures:
 *  - tap ON the fly           -> startle
 *  - 1 finger drag starting ON the fly -> panic; elsewhere -> orbit camera
 *  - 2 fingers: pinch = zoom, twist = orbit yaw
 *
 * Ray casting: screen (x,y) -> NDC -> un-project with inverse projection -> rotate into world
 * with the camera model matrix -> ray (origin = camera position, dir). Then ray-vs-sphere test.
 * All buffers are preallocated (no per-event garbage).
 */
class TouchHandler(private val density: Float) {

    interface Listener {
        fun onFlyTapped(awayX: Float, awayY: Float, awayZ: Float)
        fun onFlyDragged(awayX: Float, awayY: Float, awayZ: Float)
        fun onOrbit(dxPx: Float, dyPx: Float)
        fun onPinch(scale: Float)          // >1 = fingers spreading = zoom in
        fun onTwist(deltaRad: Float)
    }

    var listener: Listener? = null
    var camera: Camera? = null
    // fly bounding sphere, updated every frame by the renderer
    var flyX = 0f; var flyY = 0f; var flyZ = 0f; var flyRadius = 0.5f

    private val proj = DoubleArray(16)
    private val projF = FloatArray(16)
    private val invProj = FloatArray(16)
    private val model = FloatArray(16)
    private val vIn = FloatArray(4)
    private val vOut = FloatArray(4)
    private var awayX = 0f; private var awayY = 1f; private var awayZ = 0f

    private enum class Mode { NONE, UNDECIDED, ORBIT, PANIC, PINCH, IGNORE }
    private var mode = Mode.NONE
    private var prevCount = 0
    private var downX = 0f; private var downY = 0f; private var lastX = 0f; private var lastY = 0f
    private var downTime = 0L
    private var hitAtDown = false
    private var prevDist = 1f; private var prevAngle = 0f

    /** Builds the world-space ray under screen point (x,y) and tests it against the fly sphere. */
    private fun castAndPick(x: Float, y: Float, w: Int, h: Int): Boolean {
        val cam = camera ?: return false
        cam.getProjectionMatrix(proj)
        for (i in 0 until 16) projF[i] = proj[i].toFloat()
        cam.getModelMatrix(model)                       // camera -> world
        if (!Matrix.invertM(invProj, 0, projF, 0)) return false

        // screen -> NDC (y flipped)
        vIn[0] = 2f * x / w - 1f
        vIn[1] = 1f - 2f * y / h
        vIn[2] = 0.5f; vIn[3] = 1f
        Matrix.multiplyMV(vOut, 0, invProj, 0, vIn, 0)  // clip -> view space
        val iw = 1f / vOut[3]
        val vx = vOut[0] * iw; val vy = vOut[1] * iw; val vz = vOut[2] * iw

        // view-space direction -> world direction (rotation part of camera model matrix)
        var dx = model[0] * vx + model[4] * vy + model[8] * vz
        var dy = model[1] * vx + model[5] * vy + model[9] * vz
        var dz = model[2] * vx + model[6] * vy + model[10] * vz
        val l = sqrt(dx * dx + dy * dy + dz * dz)
        dx /= l; dy /= l; dz /= l
        val ox = model[12]; val oy = model[13]; val oz = model[14]

        // ray vs sphere: t = projection of (center - origin) on the ray; d2 = squared miss distance
        val cx = flyX - ox; val cy = flyY - oy; val cz = flyZ - oz
        val t = cx * dx + cy * dy + cz * dz
        val d2 = cx * cx + cy * cy + cz * cz - t * t
        val r = flyRadius * PICK_SCALE                  // generous radius so small flies are easy to hit

        // "away" direction = from closest point on the ray towards the fly (+ a bit upward)
        val qx = ox + dx * t; val qy = oy + dy * t; val qz = oz + dz * t
        var ax = flyX - qx; var ay = flyY - qy; var az = flyZ - qz
        var al = sqrt(ax * ax + ay * ay + az * az)
        if (al < 1e-3f) { ax = dx; ay = 0.3f; az = dz; al = sqrt(ax * ax + ay * ay + az * az) }
        awayX = ax / al; awayY = ay / al; awayZ = az / al

        return t > 0f && d2 <= r * r
    }

    fun onPointerEvent(ev: PointerEvent, w: Int, h: Int) {
        val ch = ev.changes
        var count = 0; var i0 = -1; var i1 = -1
        for (i in ch.indices) if (ch[i].pressed) {
            count++
            if (i0 < 0) i0 = i else if (i1 < 0) i1 = i
        }
        val slop = 12f * density

        when {
            count == 0 -> {
                if (prevCount == 1 && mode == Mode.UNDECIDED && hitAtDown)
                    listener?.onFlyTapped(awayX, awayY, awayZ)     // tap on fly only
                mode = Mode.NONE
            }
            count == 1 -> {
                val p = ch[i0].position
                if (prevCount == 0) {                              // new touch
                    mode = Mode.UNDECIDED
                    downX = p.x; downY = p.y; lastX = p.x; lastY = p.y
                    downTime = ch[i0].uptimeMillis
                    hitAtDown = castAndPick(p.x, p.y, w, h)
                } else if (prevCount >= 2) {                       // pinch -> 1 finger: ignore leftovers
                    mode = Mode.IGNORE
                } else {
                    if (mode == Mode.UNDECIDED && hypot(p.x - downX, p.y - downY) > slop)
                        mode = if (hitAtDown) Mode.PANIC else Mode.ORBIT
                    when (mode) {
                        Mode.ORBIT -> listener?.onOrbit(p.x - lastX, p.y - lastY)
                        Mode.PANIC -> { castAndPick(p.x, p.y, w, h); listener?.onFlyDragged(awayX, awayY, awayZ) }
                        else -> {}
                    }
                    lastX = p.x; lastY = p.y
                }
            }
            else -> {
                val a = ch[i0].position; val b = ch[i1].position
                val dist = hypot(b.x - a.x, b.y - a.y).coerceAtLeast(1f)
                val ang = atan2(b.y - a.y, b.x - a.x)
                if (prevCount < 2) { mode = Mode.PINCH; prevDist = dist; prevAngle = ang }
                else {
                    listener?.onPinch(dist / prevDist)
                    var d = ang - prevAngle
                    if (d > Math.PI) d -= (2 * Math.PI).toFloat()
                    if (d < -Math.PI) d += (2 * Math.PI).toFloat()
                    listener?.onTwist(d)
                    prevDist = dist; prevAngle = ang
                }
            }
        }
        prevCount = count
    }

    companion object { private const val PICK_SCALE = 1.6f }
}

/**
 * Observes touches in the INITIAL pass and does NOT consume them,
 * so it works on top of the Filament surface view.
 */
fun Modifier.flyTouch(handler: TouchHandler): Modifier = pointerInput(handler) {
    awaitPointerEventScope {
        while (true) {
            val ev = awaitPointerEvent(PointerEventPass.Initial)
            handler.onPointerEvent(ev, size.width, size.height)
        }
    }
}
