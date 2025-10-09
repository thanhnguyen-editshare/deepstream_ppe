#!/usr/bin/env python3
import sys, os, time, argparse, gi
gi.require_version("Gst", "1.0")
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib, GstRtspServer

# Constants
UNTRACKED_OBJECT_ID = 0xffffffffffffffff  # Default ID for untracked objects
TRACKED_CLASS_NAME = ["No-Helmet", "No-Vest"]  # Classes to track for duration measurement
# Alert zone format: (x, y, width, height) - center portion of the 640x640 frame
TRACKED_ZONE_ALLERT = (200, 200, 240, 240)  # Defines a rectangle in the center of the frame
TRACKED_CLASS_NAME_ZONE = ["Person"]  # Classes that trigger an alert when in the zone
Gst.init(None)


class DeepStreamVideo:

    def _handle_zone_tracking(self, class_name, track_id, current_time, obj):
        """Handle tracking and alert logic for objects in the alert zone."""
        self.tracked_objects.setdefault(track_id, {
            'class': class_name,
            'first_seen': current_time,
            'last_seen': current_time
        })
        duration = current_time - self.tracked_objects[track_id]['first_seen']
        self.tracked_objects[track_id]['last_seen'] = current_time
        obj.rect_params.border_width = 4  # Thicker border
        print(f"Class {class_name} {track_id} in restricted zone {duration}")
        if duration > 0.3:
            # Object is in the alert zone - change its display color to highlight it
            obj.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)  # Red border
            # Add a text overlay for the alert
            txt_params = obj.text_params
            txt_params.display_text = f"ALERT: {class_name} PPE violation!"
            txt_params.font_params.font_color.set(1.0, 0.0, 0.0, 1.0)  # Red text
            txt_params.font_params.font_size = 12
            txt_params.x_offset = 10
            txt_params.y_offset = 20
    """Single-video DeepStream inference using a pipeline string.

    Features:
      - Dynamic decodebin -> nvstreammux linking
      - Per-frame object count probe (optional)
            - Optional nvtracker after primary inference (enable with --tracker)
      - Verbose logging flag
    """

    def __init__(self, input_path, config_path, width=1920, height=1080, show_counts=True, verbose=False,
                 output_path=None, output_bitrate=4000000, encoder_override=None,
                 enable_rtsp=False, rtsp_port=8554, rtsp_path='ds',
                 enable_tracker=False, tracker_lib='libnvds_nvmultiobjecttracker.so',
                 tracker_config='configs/deepstream-app/tracker_config.txt',
                 tracker_width=640, tracker_height=640, tracker_past_frame=0,
                 enable_alert_zone=True, alert_zone=None, enable_second_infer=True, second_config_path=None):
        self.input_path = input_path
        self.config_path = config_path
        self.width = width
        self.height = height
        self.show_counts = show_counts
        self.verbose = verbose
        self.output_path = output_path
        self.output_bitrate = output_bitrate
        self.encoder_override = encoder_override
        self.enable_rtsp = enable_rtsp
        self.rtsp_port = rtsp_port
        self.rtsp_path = rtsp_path.strip('/') or 'ds'
        # Tracker params
        self.enable_tracker = enable_tracker
        self.tracker_lib = tracker_lib
        self.tracker_config = tracker_config
        self.tracker_width = tracker_width
        self.tracker_height = tracker_height
        self.tracker_past_frame = tracker_past_frame
        self.pipeline = None
        self.loop = None
        self.started_at = None
        self.total_counts = {}
        # Tracking data dictionaries
        self.tracked_objects = {}  # {track_id: {'class': class_name, 'first_seen': timestamp, 'last_seen': timestamp}}
        self.class_durations = {}  # {class_name: total_duration_in_seconds}
        self.rtsp_server = None
        self.rtsp_factory = None
        self.enable_second_infer = enable_second_infer
        self.second_config_path = second_config_path

    # ---------------- Internal callbacks ----------------
    def _log(self, *msg):
        if self.verbose:
            print('[VERBOSE]', *msg)

    def _on_pad_added(self, decodebin, pad, mux):
        caps = pad.get_current_caps()
        if not caps:
            return
        if 'video' in caps.to_string():
            # Use request_pad_simple (get_request_pad is deprecated)
            sinkpad = mux.request_pad_simple('sink_0')
            if sinkpad and not sinkpad.is_linked():
                self._log('Linking decodebin pad -> nvstreammux.sink_0')
                pad.link(sinkpad)

    def _osd_probe(self, pad, info, user_data):
        if not self.show_counts:
            return Gst.PadProbeReturn.OK
        try:
            import pyds
        except Exception:
            return Gst.PadProbeReturn.OK
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK
        
        # Get current timestamp
        current_time = time.time()
        
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch_meta:
            return Gst.PadProbeReturn.OK
            
        l_frame = batch_meta.frame_meta_list
        while l_frame:
            fmeta = pyds.NvDsFrameMeta.cast(l_frame.data)
            frame_counts = {}
            l_obj = fmeta.obj_meta_list
            
            # Track objects in this frame
            frame_track_ids = set()  # Keep track of IDs seen in this frame
            
            # Draw the alert zone on each frame if enabled
            if self.enable_tracker:
                # Create display meta for this frame (this holds graphical elements)
                display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                if display_meta:
                    # Define zone coordinates from instance settings
                    x, y, w, h = TRACKED_ZONE_ALLERT
                    
                    # Set up the rectangle properties
                    display_meta.num_rects = 1
                    display_meta.rect_params[0].left = x
                    display_meta.rect_params[0].top = y
                    display_meta.rect_params[0].width = w
                    display_meta.rect_params[0].height = h
                    display_meta.rect_params[0].border_width = 2
                    display_meta.rect_params[0].border_color.set(1.0, 0.0, 0.0, 1.0)  # Red border
                    display_meta.rect_params[0].has_bg_color = 1
                    display_meta.rect_params[0].bg_color.set(1.0, 0.0, 0.0, 0.2)  # Transparent red fill
                    
                    # Add the display meta to the frame
                    pyds.nvds_add_display_meta_to_frame(fmeta, display_meta)

            while l_obj:
                obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                class_name = obj.obj_label
                track_id = obj.object_id  # This is the unique tracking ID from nvtracker
                obj.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)  # Green border
                # Count objects by class in current frame
                frame_counts[class_name] = frame_counts.get(class_name, 0) + 1
                
                # Skip tracking if tracker is not enabled
                if self.enable_tracker and track_id != UNTRACKED_OBJECT_ID:
                    if class_name in TRACKED_CLASS_NAME:
                        #Track ID is valid
                        obj.rect_params.border_color.set(1.0, 1.0, 0.0, 1.0)  # Yellow border
                        frame_track_ids.add(track_id)
                        self._handle_zone_tracking(class_name, track_id, current_time, obj)
                    if class_name in TRACKED_CLASS_NAME_ZONE:
                        # Check if the object is in the alert zone
                        # Get object's coordinates and dimensions
                        bbox_left = obj.rect_params.left
                        bbox_top = obj.rect_params.top
                        bbox_width = obj.rect_params.width
                        bbox_height = obj.rect_params.height
                        
                        # Get zone coordinates from instance settings
                        zone_x, zone_y, zone_w, zone_h = TRACKED_ZONE_ALLERT
                        
                        # Calculate coordinates for the bottom line of the object
                        bottom_line_y = bbox_top + bbox_height  # Y-coordinate of the bottom line
                        left_point_x = bbox_left                # Left point X of bottom line
                        right_point_x = bbox_left + bbox_width  # Right point X of bottom line
                        
                        # Check if the bottom line's Y coordinate is within the zone's Y range
                        y_in_zone = (zone_y <= bottom_line_y <= zone_y + zone_h)

                        # Check if any part of the line's X range overlaps with the zone's X range
                        # This means checking if the horizontal line segment intersects with the zone rectangle
                        x_overlap = not (right_point_x < zone_x or left_point_x > zone_x + zone_w)

                        # The object's bottom line intersects the zone if both conditions are true
                        is_in_zone = y_in_zone and x_overlap
                        
                        if is_in_zone:
                            obj.rect_params.border_color.set(1.0, 1.0, 0.0, 1.0)  # Yellow border
                            frame_track_ids.add(track_id)
                            self._handle_zone_tracking(class_name, track_id, current_time, obj)
                            

                l_obj = l_obj.next
            
            # # Print frame counts if there are any objects
            # if frame_counts:
            #     print(f"Frame {fmeta.frame_num}: {frame_counts}")
            #     for k, v in frame_counts.items():
            #         self.total_counts[k] = self.total_counts.get(k, 0) + v
                    
            # Calculate and update durations for tracked objects that no longer appear
            if self.enable_tracker:
                self._update_tracking_durations(frame_track_ids)
                
            l_frame = l_frame.next
        return Gst.PadProbeReturn.OK

    def _update_tracking_durations(self, current_ids):
        """Update durations for tracked objects that are no longer visible."""
        current_time = time.time()
        # Look for tracked objects that are no longer in the frame
        disappeared_ids = []
        
        for track_id, data in self.tracked_objects.items():
            if track_id not in current_ids:
                # This object is no longer in the frame
                class_name = data['class']
                duration = data['last_seen'] - data['first_seen']
                
                # Add to class duration totals
                self.class_durations[class_name] = self.class_durations.get(class_name, 0) + duration
                
                if self.verbose:
                    print(f"{class_name} ID {track_id} disappeared after {duration:.2f}s")
                
                # Mark for removal from tracked objects
                disappeared_ids.append(track_id)
        
        # Remove disappeared objects from tracking
        for track_id in disappeared_ids:
            del self.tracked_objects[track_id]
        
        # Periodically report class duration totals (every ~5 seconds)
        if int(current_time) % 5 == 0:
            self._report_class_durations()

    def _on_bus_message(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print('ERROR:', err.message)
            if dbg and self.verbose:
                print('Debug:', dbg)
            loop.quit()
        elif t == Gst.MessageType.EOS:
            self._log('EOS received')
            print('EOS')
            loop.quit()
        elif t == Gst.MessageType.STATE_CHANGED and message.src == self.pipeline:
            old, new, pending = message.parse_state_changed()
            self._log(f'Pipeline state: {Gst.Element.state_get_name(old)} -> {Gst.Element.state_get_name(new)}')
        return True

    # ---------------- Public API ----------------
    def _select_encoder(self):
        """Select an available H.264 encoder in priority order.
        Returns encoder element name and property string.
        Priority (dGPU): nvh264enc -> nvv4l2h264enc -> x264enc
        Priority (Jetson/integrated): nvv4l2h264enc -> nvh264enc -> x264enc
        We detect Jetson heuristically by presence of /etc/nv_tegra_release or tegra in proc device tree.
        """
        is_jetson = os.path.isfile('/etc/nv_tegra_release') or \
            any('tegra' in open(p, 'r').read().lower() for p in [
                '/proc/device-tree/model'
            ] if os.path.isfile(p))
        if is_jetson:
            order = ['nvv4l2h264enc', 'nvh264enc', 'x264enc']
        else:
            order = ['nvh264enc', 'nvv4l2h264enc', 'x264enc']
        if self.encoder_override:
            if not Gst.ElementFactory.find(self.encoder_override):
                self._log(f"Requested encoder '{self.encoder_override}' not found; falling back to auto")
            else:
                order = [self.encoder_override] + [o for o in order if o != self.encoder_override]
        for name in order:
            if Gst.ElementFactory.find(name):
                props = ''
                if name.startswith('nvv4l2'):
                    # Jetson / V4L2 encoder expects NVMM surfaces
                    props = f" bitrate={self.output_bitrate} iframeinterval=30 insert-sps-pps=1 preset-level=1 maxperf-enable=1"
                elif name == 'nvh264enc':
                    # dGPU NVENC (may expect kbps depending on plugin build; clamp to safe range)
                    # Empirically, warning observed with large bps value; convert to kbps.
                    kbps = max(1, self.output_bitrate // 1000)
                    props = f" bitrate={kbps} rc-mode=cbr preset=low-latency-hq"
                elif name == 'x264enc':
                    # software encoder works on system memory buffers
                    kbps = max(1, self.output_bitrate // 1000)
                    props = f" bitrate={kbps} tune=zerolatency speed-preset=ultrafast key-int-max=30"
                return name, props
        # Fallback to fakesink path if no encoder
        return None, ''

    def build_pipeline(self):
        enc_name, enc_props = self._select_encoder()
        if not enc_name:
            raise RuntimeError('No suitable H.264 encoder found for file/RTSP output')

        # Build inference + optional tracker chain
        infer_chain = f"nvinfer config-file-path={self.config_path}"
        # Add second nvinfer (SGIE) if enabled
        if self.enable_second_infer and self.second_config_path:
            infer_chain += f" ! nvinfer config-file-path={self.second_config_path}"

        if self.enable_tracker:
            # nvtracker element with dynamic properties
            tracker_props = [
                f"ll-lib-file={self.tracker_lib}" if self.tracker_lib else '',
                f"ll-config-file={self.tracker_config}" if self.tracker_config else '',
                f"tracker-width={self.tracker_width}" if self.tracker_width else '',
                f"tracker-height={self.tracker_height}" if self.tracker_height else ''
            ]
            tracker_props = ' '.join(p for p in tracker_props if p)
            infer_chain += f" ! nvtracker {tracker_props}"

        # Build base up to scaled NV12 frames (after OSD)
        base = (
            f"filesrc location={self.input_path} ! decodebin name=dec "
            f"! nvstreammux name=m batch-size=1 width={self.width} height={self.height} batched-push-timeout=33000 "
            f"! {infer_chain} ! nvvideoconvert ! nvdsosd name=osd "
            f"! nvvideoconvert ! video/x-raw,width=640,height=640,format=NV12 ! tee name=t "
        )

        # File branch
        file_branch = (
            f"t. ! queue leaky=2 max-size-buffers=30 ! {enc_name}{enc_props} ! h264parse config-interval=1 ! qtmux faststart=true ! filesink location={self.output_path} sync=false "
            if self.output_path else "t. ! fakesink sync=false "
        )

        # Optional RTSP/UDP branch for direct streaming without RTSP server
        rtsp_branch = ''
        if self.enable_rtsp:
            # Direct UDP streaming with RTP payloading
            rtsp_branch = (
                f"t. ! queue max-size-buffers=120 ! {enc_name}{enc_props} ! h264parse config-interval=1 ! rtph264pay config-interval=1 pt=96 ! "
                f"udpsink host=127.0.0.1 port={self.rtsp_port} sync=true "
            )
            print(f"UDP RTP stream ready at rtp://127.0.0.1:{self.rtsp_port}")

        pipe_str = base + file_branch + rtsp_branch
        print('Pipeline string (main):\n' + pipe_str)
        self.pipeline = Gst.parse_launch(pipe_str)
        dec = self.pipeline.get_by_name('dec')
        mux = self.pipeline.get_by_name('m')
        dec.connect('pad-added', self._on_pad_added, mux)
        if self.show_counts:
            osd = self.pipeline.get_by_name('osd')
            if osd:
                sink_pad = osd.get_static_pad('sink')
                if sink_pad:
                    sink_pad.add_probe(Gst.PadProbeType.BUFFER, self._osd_probe, None)
        return self

    def start(self):
        if not self.pipeline:
            self.build_pipeline()
        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self._on_bus_message, self.loop)
        print('Starting...')
        self.started_at = time.time()
        self.pipeline.set_state(Gst.State.PLAYING)
        return self

    def _report_class_durations(self):
        """Report the total time each class has been seen in the video."""
        if self.class_durations:
            # Calculate duration for objects still being tracked
            current_time = time.time()
            for track_id, data in self.tracked_objects.items():
                class_name = data['class']
                ongoing_duration = current_time - data['first_seen']
                # Don't add to totals yet, just report
                print(f"ONGOING: {class_name} ID {track_id} present for {ongoing_duration:.2f}s")
            
            # Report duration totals
            print("----- CLASS DURATION TOTALS -----")
            for class_name, duration in self.class_durations.items():
                print(f"{class_name}: {duration:.2f} seconds total")
            print("--------------------------------")
            
    def stop(self):
        if self.pipeline:
            print('Stopping...')
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop and self.loop.is_running():
            self.loop.quit()
        if self.started_at and self.total_counts:
            dur = time.time() - self.started_at
            
            # Report counts
            if self.total_counts:
                print(f'Totals ({dur:.2f}s): {self.total_counts}')
            
            # Report final durations including current tracks
            if self.enable_tracker:
                # Update durations one last time for objects still being tracked
                current_time = time.time()
                for track_id, data in self.tracked_objects.items():
                    class_name = data['class']
                    final_duration = current_time - data['first_seen']
                    self.class_durations[class_name] = self.class_durations.get(class_name, 0) + final_duration
                
                # Final duration report
                if self.class_durations:
                    print("\n===== FINAL CLASS DURATION REPORT =====")
                    for class_name, duration in sorted(self.class_durations.items()):
                        print(f"{class_name}: {duration:.2f} seconds total")
                    print("======================================\n")
        
        return self

    def run(self):
        self.start()
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print('Interrupted')
        finally:
            self.stop()
        return 0


def parse_args(argv):
    p = argparse.ArgumentParser(description='Minimal DeepStream single-video inference')
    p.add_argument('--input', '-i', required=True, help='Input video file path')
    p.add_argument('--config', '-c', default='dstest1_pgie_config_yolo.txt', help='nvinfer config file')
    p.add_argument('--width', type=int, default=800, help='Mux width')
    p.add_argument('--height', type=int, default=800, help='Mux height')
    p.add_argument('--no-counts', action='store_true', help='Disable object count printing')
    p.add_argument('--verbose', '-v', action='store_true', help='Verbose logging (state, pad link)')
    p.add_argument('--output', '-o', default="out.mp4", help='Output MP4 file path to save 720p H.264 (if omitted, uses fakesink)')
    p.add_argument('--bitrate', type=int, default=4000000, help='H.264 encoder target bitrate (bps) for output file')
    p.add_argument('--encoder', help='Force a specific encoder (nvh264enc, nvv4l2h264enc, x264enc); auto if omitted')
    p.add_argument('--rtsp', action='store_true', help='Enable direct RTP/UDP streaming output')
    p.add_argument('--rtsp-port', type=int, default=8554, help='UDP streaming port (default 8554)')
    p.add_argument('--rtsp-path', default='ds', help='Legacy parameter, kept for compatibility')
    # Tracker related arguments
    p.add_argument('--tracker', action='store_true', help='Enable nvtracker after nvinfer')
    p.add_argument('--tracker-lib', default='/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so', help='Tracker library (ll-lib-file)')
    p.add_argument('--tracker-config', default='config_tracker_NvDCF_perf.yml', help='Tracker config file (ll-config-file)')
    p.add_argument('--tracker-width', type=int, default=800, help='Tracker input width (tracker-width)')
    p.add_argument('--tracker-height', type=int, default=800, help='Tracker input height (tracker-height)')
    p.add_argument('--tracker-past-frame', type=int, default=0, help='Enable past frame data (enable-past-frame) 0/1')
    # Second YOLO (SGIE) arguments
    p.add_argument('--second-infer',action='store_true', help='Enable second YOLO inference (SGIE)')
    p.add_argument('--second-config', default='dstest2_pgie_config_yolo.txt', help='Config file for second YOLO nvinfer (SGIE)')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    ds = DeepStreamVideo(
        input_path=args.input,
        config_path=args.config,
        width=args.width,
        height=args.height,
        show_counts=not args.no_counts,
        verbose=args.verbose,
        output_path=args.output,
        output_bitrate=args.bitrate,
        encoder_override=args.encoder,
        enable_rtsp=args.rtsp,
        rtsp_port=args.rtsp_port,
        rtsp_path=args.rtsp_path,
        enable_tracker=args.tracker,
        tracker_lib=args.tracker_lib,
        tracker_config=args.tracker_config,
        tracker_width=args.tracker_width,
        tracker_height=args.tracker_height,
        tracker_past_frame=args.tracker_past_frame,
        enable_second_infer=args.second_infer,
        second_config_path=args.second_config,
    )
    return ds.build_pipeline().run()


if __name__ == '__main__':
    sys.exit(main())