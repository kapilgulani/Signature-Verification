import React, { useState, useEffect, Fragment } from "react";
import { PhotoIcon } from "@heroicons/react/24/solid";
import { Dialog, Transition } from "@headlessui/react";
import { ExclamationTriangleIcon } from "@heroicons/react/24/outline";

const Form = () => {
    const [users, setUsers] = useState([]);
    const [selectedUser, setSelectedUser] = useState(null);
    const [genuineImagePreview, setGenuineImagePreview] = useState(null);
    const [forgedImageFile, setForgedImageFile] = useState(null);
    const [forgedImagePreview, setForgedImagePreview] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const [isDialogOpen, setIsDialogOpen] = useState(false);
    const [verificationResults, setVerificationResults] = useState({
        classification: "",
        confidence: 0,
        similarity: "",
    });

    // Fetch users from the API
    useEffect(() => {
        const fetchUsers = async () => {
            const response = await fetch("http://127.0.0.1:5000/get_users");
            const data = await response.json();
            setUsers(data.data);
            console.log(data.data);
        };
        fetchUsers();
    }, []);

    const handleUserSelection = (event) => {
        const userId = event.target.value;
        const user = users.find((user) => user.id === userId);
        setSelectedUser(user);
        if (user) {
            setGenuineImagePreview(user.signature_image);
        }
    };

    const handleForgedFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setForgedImagePreview(URL.createObjectURL(file));
            setForgedImageFile(file);
        }
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        setIsLoading(true);

        const reader = new FileReader();
        reader.readAsDataURL(forgedImageFile);
        reader.onload = async () => {
            const base64ForgedSignature = reader.result;
            try {
                const response = await fetch(
                    "http://127.0.0.1:5000/verify_signature",
                    {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            image1: selectedUser.signature_image, // Genuine image already in base64
                            image2: base64ForgedSignature, // Forged image converted to base64
                        }),
                    }
                );
                const result = await response.json();
                console.log(result);
                setVerificationResults(result); // Save the results
                setIsDialogOpen(true); // Open the dialog
            } catch (error) {
                console.error("Error submitting form:", error);
            } finally {
                setIsLoading(false);
                setGenuineImagePreview(null);
                setForgedImagePreview(null);
            }
        };
    };

    return (
        <div className="mx-auto max-w-4xl p-8">
            <div>
                <label
                    htmlFor="user-select"
                    className="block text-sm font-medium text-gray-900"
                >
                    Select User:
                </label>
                <div className="mt-1 relative">
                    <select
                        id="user-select"
                        className="appearance-none block w-full px-3 py-2 border border-gray-300 text-base rounded-md shadow-sm placeholder-gray-500 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                        onChange={handleUserSelection}
                        value={selectedUser ? selectedUser.id : ""}
                    >
                        <option value="">Select a user</option>
                        {users.map((user) => (
                            <option key={user.id} value={user.id}>
                                {user.name}
                            </option>
                        ))}
                    </select>
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                        <svg
                            className="h-4 w-4 fill-current"
                            xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 20 20"
                        >
                            <path d="M5.292 7.293a1 1 0 011.414 0L10 10.586l3.294-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" />
                        </svg>
                    </div>
                </div>
            </div>

            <form onSubmit={handleSubmit} className="space-y-8 mt-4">
                {/* Genuine Signature Display */}
                <div>
                    <label className="block text-sm font-medium leading-6 text-gray-900">
                        Genuine Signature
                    </label>
                    <div className="mt-1 flex justify-center rounded-md border-2 border-dashed border-gray-300 p-6">
                        {genuineImagePreview ? (
                            <img
                                src={genuineImagePreview}
                                alt="Genuine Signature Preview"
                                className="max-h-60"
                            />
                        ) : (
                            <div className="space-y-1 text-center">
                                <PhotoIcon className="mx-auto h-12 w-12 text-gray-400" />
                                <p className="text-sm text-gray-600">
                                    No signature loaded
                                </p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Forged Signature Upload */}
                <div>
                    <label className="block text-sm font-medium leading-6 text-gray-900">
                        Upload Forged Signature
                    </label>
                    <div className="mt-1 flex justify-center rounded-md border-2 border-dashed border-gray-300 p-6">
                        {forgedImagePreview ? (
                            <img
                                src={forgedImagePreview}
                                alt="Forged Signature Preview"
                                className="max-h-60"
                            />
                        ) : (
                            <div className="space-y-1 text-center">
                                <PhotoIcon className="mx-auto h-12 w-12 text-gray-400" />
                                <div className="flex text-sm text-gray-600">
                                    <label
                                        htmlFor="forged-signature"
                                        className="relative cursor-pointer rounded-md bg-white text-indigo-600 focus-within:outline-none focus-within:ring-2 focus-within:ring-indigo-500 focus-within:ring-offset-2 hover:text-indigo-500"
                                    >
                                        <span>Upload a file</span>
                                        <input
                                            id="forged-signature"
                                            name="forgedSignature"
                                            type="file"
                                            className="sr-only"
                                            onChange={handleForgedFileChange}
                                        />
                                    </label>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Submit Button */}
                <div>
                    <button
                        type="submit"
                        className="text-sm font-semibold leading-6 text-white bg-indigo-600 border border-transparent rounded-md shadow-sm px-4 py-2 w-full transition duration-150 ease-in-out hover:bg-indigo-500 focus:outline-none focus:border-indigo-700 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
                    >
                        {isLoading ? "Processing..." : "Verify"}
                    </button>
                </div>
            </form>

            <Transition.Root show={isDialogOpen} as={Fragment}>
                <Dialog
                    as="div"
                    className="relative z-10"
                    onClose={() => setIsDialogOpen(false)}
                >
                    <Transition.Child
                        as={Fragment}
                        enter="ease-out duration-300"
                        enterFrom="opacity-0"
                        enterTo="opacity-100"
                        leave="ease-in duration-200"
                        leaveFrom="opacity-100"
                        leaveTo="opacity-0"
                    >
                        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" />
                    </Transition.Child>

                    <div className="fixed inset-0 z-10 overflow-y-auto">
                        <div className="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
                            <Transition.Child
                                as={Fragment}
                                enter="ease-out duration-300"
                                enterFrom="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
                                enterTo="opacity-100 translate-y-0 sm:scale-100"
                                leave="ease-in duration-200"
                                leaveFrom="opacity-100 translate-y-0 sm:scale-100"
                                leaveTo="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
                            >
                                <Dialog.Panel className="relative transform overflow-hidden rounded-lg bg-white text-left shadow-xl transition-all sm:my-8 sm:w-full sm:max-w-lg">
                                    <div className="bg-white px-4 pb-4 pt-5 sm:p-6 sm:pb-4">
                                        <div className="sm:flex sm:items-start">
                                            <div className="mx-auto flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full bg-green-100 sm:mx-0 sm:h-10 sm:w-10">
                                                <ExclamationTriangleIcon
                                                    className="h-6 w-6 text-green-600"
                                                    aria-hidden="true"
                                                />
                                            </div>
                                            <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                                                <Dialog.Title
                                                    as="h3"
                                                    className="text-lg leading-6 font-medium text-gray-900"
                                                >
                                                    Verification Results
                                                </Dialog.Title>
                                                <div className="mt-2">
                                                    <p className="text-sm text-gray-500">
                                                        Similarity:{" "}
                                                        {
                                                            verificationResults.similarity
                                                        }
                                                    </p>
                                                    <p className="text-sm text-gray-500">
                                                        Classification:{" "}
                                                        {
                                                            verificationResults.classification
                                                        }
                                                    </p>
                                                    <p className="text-sm text-gray-500">
                                                        Confidence:{" "}
                                                        {verificationResults.confidence.toFixed(
                                                            2
                                                        )}
                                                    </p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="bg-gray-50 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6">
                                        <button
                                            type="button"
                                            className="inline-flex w-full justify-center rounded-md bg-blue-600 px-4 py-2 text-base font-medium text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 sm:ml-3 sm:w-auto sm:text-sm"
                                            onClick={() =>
                                                setIsDialogOpen(false)
                                            }
                                        >
                                            Close
                                        </button>
                                    </div>
                                </Dialog.Panel>
                            </Transition.Child>
                        </div>
                    </div>
                </Dialog>
            </Transition.Root>
        </div>
    );
};

export default Form;
