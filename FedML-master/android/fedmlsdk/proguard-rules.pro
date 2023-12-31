# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# If your project uses WebView with JS, uncomment the following
# and specify the fully qualified class name to the JavaScript interface
# class:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}

# Uncomment this to preserve the line number information for
# debugging stack traces.
#-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to
# hide the original source file name.
#-renamesourcefileattribute SourceFile
-ignorewarnings
-keepattributes *Annotation*
-keepattributes Exceptions
-keepattributes InnerClasses
-keepattributes Signature
-keepattributes SourceFile,LineNumberTable
-keepclasseswithmembernames class * {
    native <methods>;
}
-keep public class * extends android.app.Activity
-keep public class * extends android.app.Application
-keep public class * extends android.app.Service
-keep public class * extends android.content.BroadcastReceiver
-keep public class * extends android.content.ContentProvider
-keep public class * extends android.app.backup.BackupAgent
-keep public class * extends android.preference.Preference
-keep public class * extends android.app.Fragment
-keepattributes *Annotation*
# ==================okhttp start===================
-dontwarn okhttp3.**
-dontwarn okio.**
-dontwarn javax.annotation.**
-dontwarn org.conscrypt.**
# Animal Sniffer compileOnly dependency to ensure APIs are compatible with older versions of Java.
-dontwarn org.codehaus.mojo.animal_sniffer.*
# ==================okhttp end=====================

# ==================retrofit2 start===================
# Retain generic type information for use by reflection by converters and adapters.
-keepattributes Signature
# Retain service method parameters.
-keepclassmembernames,allowobfuscation interface * {
    @retrofit2.http.* <methods>;
}

# ==================retrofit2 end=====================

# ==================gson start=====================
-dontwarn com.google.gson.**
-keep class com.google.gson.**{*;}
-keep interface com.google.gson.**{*;}
-dontwarn sun.misc.**
-keepclassmembers,allowobfuscation class * {
  @com.google.gson.annotations.SerializedName <fields>;
}
# keep gson entity
-keep class ai.fedml.edge.service.communicator.message.**{*;}
-keep class ai.fedml.edge.request.**{*;}
-keep class ai.fedml.edge.request.parameter.**{*;}
-keep class ai.fedml.edge.request.response.**{*;}
-keep class ai.fedml.edge.nativemnn.**{*;}
-keep class ai.fedml.edge.utils.**{*;}
# ==================gson end=====================

-keep interface ai.fedml.edge.FedEdgeApi{*;}
-keep interface ai.fedml.edge.OnTrainingStatusListener{*;}
-keep interface ai.fedml.edge.OnTrainProgressListener{*;}
-keep interface ai.fedml.edge.nativemobilenn.TrainingCallback{*;}
-keep class ai.fedml.edge.FedEdgeManager {
    public *;
}
