; Auto-generated. Do not edit!


(cl:in-package lab2-msg)


;//! \htmlinclude NavTargetFeedback.msg.html

(cl:defclass <NavTargetFeedback> (roslisp-msg-protocol:ros-message)
  ((distance
    :reader distance
    :initarg :distance
    :type std_msgs-msg:Float32
    :initform (cl:make-instance 'std_msgs-msg:Float32)))
)

(cl:defclass NavTargetFeedback (<NavTargetFeedback>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <NavTargetFeedback>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'NavTargetFeedback)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name lab2-msg:<NavTargetFeedback> is deprecated: use lab2-msg:NavTargetFeedback instead.")))

(cl:ensure-generic-function 'distance-val :lambda-list '(m))
(cl:defmethod distance-val ((m <NavTargetFeedback>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader lab2-msg:distance-val is deprecated.  Use lab2-msg:distance instead.")
  (distance m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <NavTargetFeedback>) ostream)
  "Serializes a message object of type '<NavTargetFeedback>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'distance) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <NavTargetFeedback>) istream)
  "Deserializes a message object of type '<NavTargetFeedback>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'distance) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<NavTargetFeedback>)))
  "Returns string type for a message object of type '<NavTargetFeedback>"
  "lab2/NavTargetFeedback")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'NavTargetFeedback)))
  "Returns string type for a message object of type 'NavTargetFeedback"
  "lab2/NavTargetFeedback")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<NavTargetFeedback>)))
  "Returns md5sum for a message object of type '<NavTargetFeedback>"
  "e1093a895d389825abe211cea01772a0")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'NavTargetFeedback)))
  "Returns md5sum for a message object of type 'NavTargetFeedback"
  "e1093a895d389825abe211cea01772a0")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<NavTargetFeedback>)))
  "Returns full string definition for message of type '<NavTargetFeedback>"
  (cl:format cl:nil "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%std_msgs/Float32 distance~%~%~%================================================================================~%MSG: std_msgs/Float32~%float32 data~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'NavTargetFeedback)))
  "Returns full string definition for message of type 'NavTargetFeedback"
  (cl:format cl:nil "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======~%std_msgs/Float32 distance~%~%~%================================================================================~%MSG: std_msgs/Float32~%float32 data~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <NavTargetFeedback>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'distance))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <NavTargetFeedback>))
  "Converts a ROS message object to a list"
  (cl:list 'NavTargetFeedback
    (cl:cons ':distance (distance msg))
))
