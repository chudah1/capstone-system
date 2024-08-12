import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:video_player/video_player.dart';
import 'firebase_options.dart';
import 'dart:html' as html;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Video Processing App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  bool _isUploading = false;
  bool _isProcessing = false;
  String _responseMessage = '';
  VideoPlayerController? _controller;
  Uint8List? _videoBytes;

  Future<void> _uploadVideo() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.video,
    );

    if (result != null && result.files.single.bytes != null) {
      setState(() {
        _videoBytes = result.files.single.bytes!;
        _responseMessage = '';

        html.Blob blob = html.Blob([_videoBytes!]);
        String url = html.Url.createObjectUrlFromBlob(blob);

        _controller = VideoPlayerController.networkUrl(Uri.parse(url))
          ..initialize().then((_) {
            setState(() {});
            _controller!.play();
          });
      });
    } else {
      setState(() {
        _responseMessage = 'No video selected.';
      });
    }
  }

  Future<void> _processVideo() async {
    if (_videoBytes == null) {
      setState(() {
        _responseMessage = 'No video to process. Please upload a video first.';
      });
      return;
    }

    setState(() {
      _isProcessing = true;
      _responseMessage = '';
    });

    final request = http.MultipartRequest(
      'POST',
      Uri.parse('http://34.40.142.212:5000/process_video'),
    );
    request.files.add(
      http.MultipartFile.fromBytes(
        'file',
        _videoBytes!,
        filename: 'uploaded_video.mp4',
      ),
    );

    final response = await request.send();

    if (response.statusCode == 202) {
      setState(() {
        _responseMessage = 'File uploaded successfully. Processing started.';
      });

      FirebaseFirestore.instance
          .collection('transactions')
          .snapshots()
          .listen((snapshot) {
        if (snapshot.docs.isNotEmpty) {
          setState(() {
            _isProcessing = false;
            _responseMessage = 'Video processed successfully.';
          });
        }
      });
    } else {
      setState(() {
        _responseMessage = 'Failed to process video.';
        _isProcessing = false;
      });
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AI recording App'),
      ),
      body: Row(
        children: [
          Expanded(
            flex: 1,
            child: SingleChildScrollView(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  ElevatedButton(
                    onPressed: _isUploading ? null : _uploadVideo,
                    child: _isUploading
                        ? const CircularProgressIndicator()
                        : const Text('Upload Video'),
                  ),
                  const SizedBox(height: 20),
                  ElevatedButton(
                    onPressed: _isProcessing ? null : _processVideo,
                    child: _isProcessing
                        ? const CircularProgressIndicator()
                        : const Text('Process Video'),
                  ),
                  const SizedBox(height: 20),
                  if (_responseMessage.isNotEmpty) Text(_responseMessage),
                  const SizedBox(height: 20),
                  if (_controller != null && _controller!.value.isInitialized)
                    Container(
                      width: 900,
                      height: 700,
                      child: VideoPlayer(_controller!),
                    ),
                  if (_controller != null && _controller!.value.isInitialized)
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        IconButton(
                          icon: Icon(
                            _controller!.value.isPlaying
                                ? Icons.pause
                                : Icons.play_arrow,
                          ),
                          onPressed: () {
                            setState(() {
                              _controller!.value.isPlaying
                                  ? _controller!.pause()
                                  : _controller!.play();
                            });
                          },
                        ),
                      ],
                    ),
                ],
              ),
            ),
          ),
          const Expanded(
            flex: 1,
            child: TransactionsTable(),
          ),
        ],
      ),
    );
  }
}

class TransactionsTable extends StatelessWidget {
  const TransactionsTable({super.key});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder(
      stream: FirebaseFirestore.instance.collection('transactions').snapshots(),
      builder: (context, AsyncSnapshot<QuerySnapshot> snapshot) {
        if (!snapshot.hasData) {
          return const Center(child: CircularProgressIndicator());
        }

        return DataTable(
          columns: const [
            DataColumn(label: Text('Buyer')),
            DataColumn(label: Text('Cashier')),
            DataColumn(label: Text('Product')),
          ],
          rows: snapshot.data!.docs.map((doc) {
            return DataRow(
              cells: [
                DataCell(Text(doc['buyer'])),
                DataCell(Text(doc['cashier'])),
                DataCell(Text(doc['product'])),
              ],
            );
          }).toList(),
        );
      },
    );
  }
}
